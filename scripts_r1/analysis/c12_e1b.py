"""
C12 E1b — reference models trained WITH IoMT traffic (plan reports/comment12/e1b_plan.md, B1-B8 approved 2026-10-30).

  python c12_e1b.py --repo <repo> --cicids-cache <dir with prepared_*.npz> --train-work <work_c12/e1b_train>
                    --test-work <work_c12/e1> --label-map-train label_map_train.json --label-map-test label_map.json
                    --extractor e0_extractor_config.json --out <dir> --models <dir> [--trials 50] [--only M1,M2]

M0  deployed v2 model (edge/models_v2/lgbm_classifier.pkl), re-evaluated through the same code path (sanity check
    against E1 and the CICIDS regression lock)
M1  IoMT only: CICIoMT2024 train captures (BENIGN = Benign_train; DOS; PORT_SCAN)
M2  joint: CICIDS-2017 v2 train/val/tune (same split indices as the deployed model) + CICIoMT2024 train
Protocol (v2, Comment 8): exact-vector dedup with label-conflict drop (IoMT pool), stratified split of the IoMT TRAIN
captures into train/val/tune (60/10/15 renormalised), the CICIoMT2024 TEST captures are the test set (CIC's capture-level
separation), train-only balancing (balance_classes), seeded TPE (50 trials) via train_pipeline_v2.fit_lightgbm with a
per-model SQLite study (resumable), refit with the best trial's early-stopping iteration, tau cost-optimal on the
model's own tune split (r = 10), and results at tau = 0.25.
Deviation (documented): +/-inf in Flow Packets/s and Flow Bytes/s use the deployed v2 train-only values for every model
(c12_common.to_x33), so all models see the deployed preprocessing.
Nothing is written to the repository; models go to --models.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c12_common import apply_variant, clopper_pearson, say, to_x33  # noqa: E402
from c12_e1 import labelled  # noqa: E402

R_COST = 10.0


def load_iomt(work: Path, lm: dict, ex: dict, repo: Path) -> dict:
    """Per capture: label, map to the 33 features, keep compact arrays (memory)."""
    from edge.training.label_mapper import CLASS_NAMES
    Xs, ys, labs, caps, ws, bens = [], [], [], [], [], []
    for pkl in sorted(work.glob("*.pkl")):
        df = labelled(work, lm, [pkl])
        if not len(df):
            continue
        df = apply_variant(df, ex)
        X = to_x33(df, repo)
        tc = df["true_class"].to_numpy()
        keep = np.array([isinstance(t, str) for t in tc])           # drops "no SAINT class" (MQTT-Malformed, B7)
        Xs.append(X[keep]); ys.append(np.array([CLASS_NAMES.index(t) for t in tc[keep]], dtype=np.int64))
        labs.append(df["label"].to_numpy()[keep]); caps.append(df["capture"].to_numpy()[keep])
        ws.append(df["weight"].to_numpy()[keep]); bens.append(df["capture_seconds"].to_numpy()[keep])
        say(f"  {pkl.name[:70]}: {keep.sum():,} flows ({(~keep).sum():,} without SAINT class)")
    return {"X": np.concatenate(Xs), "y": np.concatenate(ys), "label": np.concatenate(labs),
            "capture": np.concatenate(caps), "weight": np.concatenate(ws), "capture_seconds": np.concatenate(bens)}


def load_test(work: Path, lm: dict, ex: dict, repo: Path) -> dict:
    """The E1 test set (attack test captures + Benign_test + profiling), all labels kept (incl. no-class)."""
    from edge.training.label_mapper import CLASS_NAMES
    parts = {k: [] for k in ("X", "label", "true", "capture", "weight", "is_attack", "capture_seconds")}
    for pkl in sorted(work.glob("*.pkl")):
        df = labelled(work, lm, [pkl])
        if not len(df):
            continue
        df = apply_variant(df, ex)
        parts["X"].append(to_x33(df, repo)); parts["label"].append(df["label"].to_numpy())
        parts["true"].append(np.array([CLASS_NAMES.index(t) if isinstance(t, str) else -1 for t in df["true_class"]]))
        parts["capture"].append(df["capture"].to_numpy()); parts["weight"].append(df["weight"].to_numpy())
        parts["is_attack"].append(df["is_attack"].to_numpy(bool)); parts["capture_seconds"].append(df["capture_seconds"].to_numpy())
    return {k: np.concatenate(v) for k, v in parts.items()}


def load_cicids(cache: Path, repo: Path) -> dict:
    committed = np.load(repo / "edge/models_v2/split_indices_v2.npz")
    for c in sorted(Path(cache).glob("prepared_*.npz")):
        z = np.load(c)
        if all(np.array_equal(z[f"part_{k}"], committed[k]) for k in ("train", "val", "tune", "test")):
            return {"X": z["X"].astype(np.float32), "y": z["y"].astype(np.int64),
                    "parts": {k: committed[k] for k in ("train", "val", "tune", "test")}, "cache": str(c)}
    raise SystemExit("no CICIDS prepared cache matches the committed v2 split")


def split3(y: np.ndarray, seed: int = 42) -> dict:
    """Stratified train/val/tune = 60/10/15 renormalised (the test set is the CICIoMT2024 TEST captures)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    rest, tune = train_test_split(idx, test_size=15 / 85, stratify=y, random_state=seed)
    train, val = train_test_split(rest, test_size=10 / 70, stratify=y[rest], random_state=seed)
    return {"train": np.sort(train), "val": np.sort(val), "tune": np.sort(tune)}


def s_attack(model, X: np.ndarray) -> tuple:
    """1 - p_BENIGN and the top attack class (argmax over the non-BENIGN classes the model knows)."""
    proba = model.predict_proba(X)
    cls = list(model.classes_)
    pb = proba[:, cls.index(0)] if 0 in cls else np.zeros(len(X))
    att = [i for i, c in enumerate(cls) if c != 0]
    top = np.array(cls)[np.array(att)[np.argmax(proba[:, att], axis=1)]]
    return 1.0 - pb, top


def cost_optimal_tau(s: np.ndarray, y: np.ndarray) -> dict:
    """tau minimising r*FN + FP on the tune split (same rule as the deployed tau)."""
    atk = y != 0
    grid = np.unique(np.concatenate([np.linspace(0.001, 0.999, 999), [0.25]]))
    best = None
    for t in grid:
        al = s >= t
        fn = int((atk & ~al).sum()); fp = int((~atk & al).sum())
        c = R_COST * fn + fp
        if best is None or c < best[0]:
            best = (c, float(t), fn, fp)
    return {"tau": best[1], "cost": best[0], "tune_fn": best[2], "tune_fp": best[3], "r": R_COST}


def eval_iomt(s, top, T, tau) -> dict:
    al = s >= tau
    ben, atk = ~T["is_attack"], T["is_attack"]
    from edge.training.label_mapper import CLASS_NAMES
    hours = float(pd.Series(T["capture_seconds"][ben], index=T["capture"][ben]).groupby(level=0).first().sum() / 3600)
    k, n = int(al[ben].sum()), int(ben.sum())
    lo, hi = clopper_pearson(k, n)
    out = {"tau": tau, "benign": {"flows": n, "alerts": k, "rate": k / n if n else None, "ci95": [lo, hi],
                                  "hours": hours, "per_hour": k / hours if hours else None},
           "attack": {"flows": int(atk.sum()), "detected": int((al & atk).sum()),
                      "detected_weighted": float((T["weight"] * al)[atk].sum() / T["weight"][atk].sum())},
           "per_label": {}, "runs_detected": 0, "runs": 0}
    named = al & atk & (T["true"] >= 0)
    out["attack"]["naming"] = {"k": int((top[named] == T["true"][named]).sum()), "n": int(named.sum())}
    for lab in sorted(set(T["label"][atk])):
        m = atk & (T["label"] == lab)
        d = m & al
        tc = T["true"][m][0]
        out["per_label"][str(lab)] = {
            "flows": int(m.sum()), "detected_pct": 100 * float(al[m].mean()),
            "naming_pct": 100 * float((top[d] == tc).mean()) if d.any() and tc >= 0 else None,
            "named_as": {CLASS_NAMES[c]: int(v) for c, v in zip(*np.unique(top[d], return_counts=True))} if d.any() else {}}
        out["runs"] += 1; out["runs_detected"] += int(d.any())
    return out


def eval_cicids(model, X, y, tau) -> dict:
    s, top = s_attack(model, X)
    al = s >= tau; atk = y != 0
    return {"tau": tau, "TP": int((al & atk).sum()), "FP": int((al & ~atk).sum()), "TN": int((~al & ~atk).sum()),
            "FN": int((~al & atk).sum())}


def train_model(name, Xtr, ytr, Xva, yva, cfg, trials, models_dir: Path) -> tuple:
    import joblib

    from edge.training.train_pipeline import balance_classes
    from edge.training.train_pipeline_v2 import fit_lightgbm
    pkl = models_dir / f"{name}.pkl"
    info_p = models_dir / f"{name}_training.json"
    if pkl.exists() and info_p.exists():
        say(f"{name}: cached model {pkl}")
        return joblib.load(pkl), json.loads(info_p.read_text())
    b = cfg["balancing"]
    try:
        Xb, yb = balance_classes(Xtr, ytr, b["max_benign_multiplier"], b["smote_threshold"], b["random_state"])
        bal_note = "balance_classes (train only)"
    except ValueError as e:                     # only on degenerate (smoke-test) data
        Xb, yb, bal_note = Xtr, ytr, f"balancing skipped: {e}"
        say(f"{name}: {bal_note}")
    c = dict(cfg); c["protocol_id"] = f"e1b_{name}"
    t0 = time.time()
    say(f"{name}: training on {len(yb):,} rows (balanced from {len(ytr):,}), val {len(yva):,}, {trials} trials")
    lg = fit_lightgbm(Xb, yb, Xva, yva, c, trials, record_curves=False,
                      storage=f"sqlite:///{(models_dir / f'optuna_{name}.db').resolve()}")
    model = lg.pop("model")
    info = {k: v for k, v in lg.items() if k not in ("optuna_trials",)}
    info["n_trials_completed"] = len(lg.get("optuna_trials", []))
    info["balancing"] = bal_note; info["rows_train_balanced"] = int(len(yb)); info["rows_train"] = int(len(ytr)); info["rows_val"] = int(len(yva))
    info["classes"] = [int(x) for x in model.classes_]; info["seconds"] = round(time.time() - t0, 1)
    joblib.dump(model, pkl); info_p.write_text(json.dumps(info, indent=1, default=str))
    say(f"{name}: done in {info['seconds']:.0f} s; refit check {info.get('refit_check')}")
    return model, info


def main():
    ap = argparse.ArgumentParser()
    for k in ("repo", "cicids-cache", "train-work", "test-work", "label-map-train", "label-map-test", "extractor", "out", "models"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--trials", type=int, default=50); ap.add_argument("--only", default="M1,M2")
    ap.add_argument("--smoke-cicids", type=int, default=0, help="laptop smoke test only: subsample each CICIDS part")
    a = ap.parse_args()
    repo = Path(a.repo).resolve(); sys.path.insert(0, str(repo))
    out, models_dir = Path(a.out), Path(a.models); out.mkdir(parents=True, exist_ok=True); models_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    from edge.training.dedup import cross_partition_overlap, deduplicate, row_hashes
    from edge.training.label_mapper import CLASS_NAMES
    from edge.training.train_pipeline_v2 import class_counts, load_config
    cfg = load_config(repo / "edge/training/configs/v2_dedup.json")
    ex = json.loads(Path(a.extractor).read_text())
    res = {"repo_commit": os.popen(f"git -C {repo} rev-parse HEAD").read().strip(), "protocol": cfg["protocol_id"],
           "trials": a.trials, "r_cost": R_COST, "extractor": ex}

    say("IoMT TRAIN pool (CICIoMT2024 train captures)")
    P = load_iomt(Path(a.train_work), json.loads(Path(a.label_map_train).read_text()), ex, repo)
    res["iomt_pool_raw"] = {"rows": int(len(P["y"])), "class_counts": class_counts(P["y"]),
                            "by_label": pd.Series(P["label"]).value_counts().to_dict()}
    keep, dd = deduplicate(P["X"], P["y"], cfg["dedup"]["conflict_policy"])
    for k in P:
        P[k] = P[k][keep]
    res["iomt_pool_dedup"] = {"report": dd, "rows": int(len(P["y"])), "class_counts": class_counts(P["y"])}
    S = split3(P["y"])
    res["iomt_split"] = {k: {"rows": int(len(v)), "class_counts": class_counts(P["y"][v])} for k, v in S.items()}
    say(f"IoMT pool after dedup: {res['iomt_pool_dedup']['class_counts']}")

    say("TEST set (CICIoMT2024 test captures + Benign_test + profiling = the E1 test set)")
    T = load_test(Path(a.test_work), json.loads(Path(a.label_map_test).read_text()), ex, repo)
    # leakage report: test vectors identical to an IoMT train/val/tune vector (CIC split by capture, not by flow)
    hp = set(row_hashes(P["X"]).tolist()); ht = row_hashes(T["X"])
    dup = np.array([h in hp for h in ht.tolist()])
    res["test_rows_identical_to_iomt_training_vectors"] = {
        "attack": int((dup & T["is_attack"]).sum()), "benign": int((dup & ~T["is_attack"]).sum()),
        "attack_flows": int(T["is_attack"].sum()), "benign_flows": int((~T["is_attack"]).sum())}

    say("CICIDS-2017 v2 prepared data (same split indices as the deployed model)")
    C = load_cicids(Path(a.cicids_cache), repo)
    res["cicids_cache"] = C["cache"]; Cp = C["parts"]
    if a.smoke_cicids:
        rng = np.random.default_rng(0)
        Cp = {k: np.sort(rng.choice(v, min(a.smoke_cicids, len(v)), replace=False)) for k, v in Cp.items()}
        res["SMOKE_TEST_cicids_subsample"] = a.smoke_cicids

    models = {"M0": joblib.load(repo / "edge/models_v2/lgbm_classifier.pkl")}
    taus = {"M0": 0.25}
    if "M1" in a.only:
        m, info = train_model("M1", P["X"][S["train"]], P["y"][S["train"]], P["X"][S["val"]], P["y"][S["val"]],
                              cfg, a.trials, models_dir)
        models["M1"] = m; res["M1_training"] = info
        s_t, _ = s_attack(m, P["X"][S["tune"]]); taus["M1"] = cost_optimal_tau(s_t, P["y"][S["tune"]])["tau"]
        res["M1_tau"] = cost_optimal_tau(s_t, P["y"][S["tune"]])
    if "M2" in a.only:
        cross = cross_partition_overlap(np.concatenate([C["X"], P["X"]]), np.arange(len(C["y"])),
                                        np.arange(len(C["y"]), len(C["y"]) + len(P["y"])))
        res["M2_iomt_rows_identical_to_a_cicids_vector"] = int(cross)
        Xtr = np.concatenate([C["X"][Cp["train"]], P["X"][S["train"]]]); ytr = np.concatenate([C["y"][Cp["train"]], P["y"][S["train"]]])
        Xva = np.concatenate([C["X"][Cp["val"]], P["X"][S["val"]]]); yva = np.concatenate([C["y"][Cp["val"]], P["y"][S["val"]]])
        m, info = train_model("M2", Xtr, ytr, Xva, yva, cfg, a.trials, models_dir)
        models["M2"] = m; res["M2_training"] = info
        Xtu = np.concatenate([C["X"][Cp["tune"]], P["X"][S["tune"]]]); ytu = np.concatenate([C["y"][Cp["tune"]], P["y"][S["tune"]]])
        s_t, _ = s_attack(m, Xtu); res["M2_tau"] = cost_optimal_tau(s_t, ytu); taus["M2"] = res["M2_tau"]["tau"]

    res["evaluation"] = {}
    for name, m in models.items():
        s, top = s_attack(m, T["X"])
        ev = {"iomt_test_at_own_tau": eval_iomt(s, top, T, taus[name])}
        if taus[name] != 0.25:
            ev["iomt_test_at_0.25"] = eval_iomt(s, top, T, 0.25)
        ev["cicids_v2_test_at_own_tau"] = eval_cicids(m, C["X"][Cp["test"]], C["y"][Cp["test"]], taus[name])
        if taus[name] != 0.25:
            ev["cicids_v2_test_at_0.25"] = eval_cicids(m, C["X"][Cp["test"]], C["y"][Cp["test"]], 0.25)
        ev["classes"] = [CLASS_NAMES[int(c)] for c in m.classes_]
        # V1 / v2 validation gates on the IoMT test flows that have a SAINT class (argmax classifier),
        # recall gates only for classes present in that test set
        from edge.training.train_pipeline_v2 import evaluate_classifier
        cm = T["true"] >= 0
        met = evaluate_classifier(m, T["X"][cm], T["true"][cm])
        present = [c for c in ("PORT_SCAN", "BRUTE_FORCE", "DOS") if met["support_per_class"][c] > 0]
        ev["gates_iomt_test"] = {"f1_weighted": met["f1_weighted"], "f1_weighted_gt_0_95": met["f1_weighted"] > 0.95,
                                 "fpr": met["false_positive_rate"], "fpr_lt_0_02": (met["false_positive_rate"] or 0) < 0.02,
                                 **{f"{c.lower()}_recall": met["recall_per_class"][c] for c in present},
                                 **{f"{c.lower()}_recall_gt_0_90": met["recall_per_class"][c] > 0.90 for c in present}}
        res["evaluation"][name] = ev
        e = ev["iomt_test_at_own_tau"]
        say(f"{name} (tau {taus[name]:.3f}): benign {e['benign']['alerts']}/{e['benign']['flows']} "
            f"({e['benign']['per_hour']:.3f}/h), attack runs {e['runs_detected']}/{e['runs']}, "
            f"flows {e['attack']['detected']:,}/{e['attack']['flows']:,}, naming {e['attack']['naming']}; "
            f"CICIDS test {ev['cicids_v2_test_at_own_tau']}")
    (out / "e1b.json").write_text(json.dumps(res, indent=1, default=str))
    say("written e1b.json")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
