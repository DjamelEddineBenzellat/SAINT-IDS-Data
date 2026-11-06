"""
C8 — Step 1 evidence for Reviewer 1 Comment 4 (Isolation Forest) and the SHAP cost.
Plan: reports/comment4/step1_plan.md. Read-only w.r.t. the repository.

  python c8_step1_evidence.py --repo <repo> --cache <dir with prepared_*.npz> --out <dir> --stage e1|e2|e3

E1  IF ablation on the v2 tune/test splits: IF standalone; C′ vs C′ OR IF flag; complementarity on
    C′ misses; IF score AUROC; IF-score threshold for a fixed extra-false-alarm budget chosen on tune.
E2  Novelty (unseen attack class): for each attack class k, fit LightGBM (v2 best params, no Optuna) on
    train WITHOUT class k with early stopping on val WITHOUT k; on test rows of class k compare
    s_attack >= τ, the IF flag and their union, with the benign false-alarm cost (validity: <= 1 %).
E3  SHAP methods on 1,000 v2 test flows (500 alerting + 500 non-alerting, seed 0), one core:
    M0 shap TreeExplainer(model, data=bg100) [current], M1 shap tree_path_dependent,
    M2 LightGBM booster.predict(pred_contrib=True). Latency, additivity failures, top-5 agreement vs M0.
Each stage writes <out>/<stage>.json; an existing file is not recomputed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_T0 = time.time()
ROOT = Path(".")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.0f}s] {msg}", flush=True)


def pin(cores):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cores)


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def load_data(cache_dir):
    committed = np.load(ROOT / "edge/models_v2/split_indices_v2.npz")
    for c in sorted(Path(cache_dir).glob("prepared_*.npz")):
        z = np.load(c)
        if all(np.array_equal(z[f"part_{k}"], committed[k]) for k in ("train", "val", "tune", "test")):
            say(f"data: {c.name} matches the committed v2 split")
            return z, {k: committed[k] for k in ("train", "val", "tune", "test")}
    raise SystemExit(f"no prepared_*.npz in {cache_dir} matches the committed v2 split")


def onnx_scores(X):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ROOT / "edge/models_v2/lgbm_classifier.onnx"), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    P = np.zeros((len(X), 6))
    for a in range(0, len(X), 50000):
        for r, d in enumerate(sess.run(None, {name: X[a:a + 50000].astype(np.float32)})[1]):
            for k, v in d.items():
                P[a + r, int(k)] = v
    return 1.0 - P[:, 0]


def auroc(score, pos):
    from sklearn.metrics import average_precision_score, roc_auc_score
    return {"auroc": float(roc_auc_score(pos, score)), "average_precision": float(average_precision_score(pos, score))}


def counts(alert, y):
    atk = y != 0
    tp, fp = int((alert & atk).sum()), int((alert & ~atk).sum())
    fn, tn = int((~alert & atk).sum()), int((~alert & ~atk).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "recall": rec, "precision": prec,
            "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "benign_alert_pct": 100 * fp / (fp + tn), "benign_alert_CI95_pct": [100 * v for v in wilson(fp, fp + tn)]}


def stage_e1(z, P, tau, names):
    import joblib
    iforest = joblib.load(ROOT / "edge/models_v2/isolation_forest.pkl"); iforest.set_params(n_jobs=-1)
    out = {"tau": tau}
    data = {}
    for part in ("tune", "test"):
        X, y = z["X"][P[part]], z["y"][P[part]]
        say(f"E1 {part}: ONNX scores"); s = onnx_scores(X)
        say(f"E1 {part}: IF predict + score"); t = time.time()
        if_flag = iforest.predict(X) == -1
        if_score = -iforest.decision_function(X)          # higher = more anomalous
        say(f"E1 {part}: IF batch time {time.time() - t:.1f}s")
        data[part] = (X, y, s, if_flag, if_score)
        cprime = s >= tau
        union = cprime | if_flag
        atk = y != 0
        missed = atk & ~cprime
        r = {"n": int(len(y)),
             "if_standalone": counts(if_flag, y), "cprime": counts(cprime, y), "cprime_or_if": counts(union, y),
             "if_score_ranking": auroc(if_score, atk), "s_attack_ranking": auroc(s, atk),
             "cprime_misses": int(missed.sum()), "cprime_misses_flagged_by_if": int((missed & if_flag).sum()),
             "added_by_if_by_class": {names[k]: int((union & ~cprime & (y == k)).sum()) for k in range(1, 6)},
             "added_false_alarms": int((union & ~cprime & ~atk).sum()),
             "misses_by_class": {names[k]: [int((missed & (y == k)).sum()), int((missed & if_flag & (y == k)).sum())]
                                 for k in range(1, 6)}}
        out[part] = r
        say(f"E1 {part}: C′ misses {r['cprime_misses']}, flagged by IF {r['cprime_misses_flagged_by_if']}, "
            f"IF adds FP {r['added_false_alarms']}; IF standalone F1 {r['if_standalone']['F1']:.4f}")
    # Fair chance for the IF: choose an IF-score threshold on TUNE that adds at most +0.05 / +0.1 / +0.5 pp
    # benign alerts on top of C′, then apply it once to TEST.
    Xt, yt, st, _, it = data["tune"]; Xs, ys, ss, _, is_ = data["test"]
    ben_t = (yt == 0) & (st < tau); n_ben_t = int((yt == 0).sum())
    out["if_threshold_budget"] = {}
    for budget_pp in (0.05, 0.1, 0.5):
        k = int(np.floor(budget_pp / 100 * n_ben_t))
        cand = np.sort(it[ben_t])[::-1]
        thr = float(cand[k]) if k < len(cand) else float(cand[-1])
        res = {}
        for part, (y_, s_, i_) in (("tune", (yt, st, it)), ("test", (ys, ss, is_))):
            alert = (s_ >= tau) | (i_ > thr)
            base = s_ >= tau
            res[part] = {"extra_TP": int((alert & ~base & (y_ != 0)).sum()), "extra_FP": int((alert & ~base & (y_ == 0)).sum()),
                         "with_if": counts(alert, y_)}
        out["if_threshold_budget"][f"+{budget_pp}pp"] = {"if_score_threshold": thr, **res}
        say(f"E1 budget +{budget_pp}pp: test extra TP {res['test']['extra_TP']}, extra FP {res['test']['extra_FP']}")
    return out


def stage_e2(z, P, tau, names):
    """
    Unseen-class test. For each attack class k: fit LightGBM with the v2 best hyperparameters on TRAIN
    without k, with early stopping (30 rounds, up to the searched n_estimators) on VAL without k — the
    same procedure as the v2 Optuna trials. A fixed tree count diverged in the first E2 run
    (BRUTE_FORCE left out: 59.6 % benign alerts). Reference run "none" = no class removed.
    A refit is marked invalid if its benign alert rate on test exceeds 1 %.
    """
    import joblib
    import lightgbm as lgb
    rep = json.loads((ROOT / "edge/models_v2/training_report.json").read_text())
    params = dict(rep["lightgbm"]["lgbm_params"]); params["n_jobs"] = -1
    params["n_estimators"] = int(rep["optuna_best_params"]["n_estimators"])     # searched value (441)
    iforest = joblib.load(ROOT / "edge/models_v2/isolation_forest.pkl"); iforest.set_params(n_jobs=-1)
    Xtr, ytr = z["Xtr"], z["ytr"]; Xva, yva = z["X"][P["val"]], z["y"][P["val"]]
    Xs, ys = z["X"][P["test"]], z["y"][P["test"]]
    say("E2: IF flags on test"); if_flag = iforest.predict(Xs) == -1
    if_score = -iforest.decision_function(Xs)
    ben = ys == 0
    out = {"tau": tau, "params": params, "procedure": "train minus k, early stopping on val minus k (30 rounds)",
           "validity_rule": "benign alert rate on test <= 1 %",
           "if_benign_false_alarm_pct": 100 * float(if_flag[ben].mean())}
    for k in [None, 1, 2, 3, 4, 5]:
        tag = "none" if k is None else names[k]
        kt = np.ones(len(ytr), bool) if k is None else ytr != k
        kv = np.ones(len(yva), bool) if k is None else yva != k
        say(f"E2: fit without {tag} ({int((~kt).sum())} train rows removed)")
        t = time.time()
        m = lgb.LGBMClassifier(**params).fit(Xtr[kt], ytr[kt], eval_set=[(Xva[kv], yva[kv])],
                                             callbacks=[lgb.early_stopping(30, verbose=False)])
        fit_s = time.time() - t
        cls = list(m.classes_); assert cls[0] == 0
        s = 1 - m.predict_proba(Xs)[:, 0]
        det = s >= tau; union = det | if_flag
        r = {"fit_seconds": round(fit_s, 1), "best_iteration": int(m.best_iteration_ or params["n_estimators"]),
             "benign_alert_pct_s_attack": 100 * float(det[ben].mean()),
             "benign_alert_pct_union": 100 * float(union[ben].mean())}
        r["valid"] = r["benign_alert_pct_s_attack"] <= 1.0
        if k is not None:
            hk = ys == k
            r.update({"heldout_test_rows": int(hk.sum()),
                      "detected_by_s_attack": int(det[hk].sum()), "detected_by_if": int(if_flag[hk].sum()),
                      "detected_by_union": int(union[hk].sum()),
                      "recall_s_attack": float(det[hk].mean()), "recall_s_attack_CI95": wilson(int(det[hk].sum()), int(hk.sum())),
                      "recall_if": float(if_flag[hk].mean()), "recall_if_CI95": wilson(int(if_flag[hk].sum()), int(hk.sum())),
                      "recall_union": float(union[hk].mean()), "added_by_if": int((union & ~det)[hk].sum()),
                      "ranking_vs_benign_s_attack": auroc(s[hk | ben], hk[hk | ben]),
                      "ranking_vs_benign_if_score": auroc(if_score[hk | ben], hk[hk | ben]),
                      "heldout_predicted_as": {names[c]: int((m.predict(Xs[hk]) == c).sum()) for c in cls}})
            msg = (f"unseen-class recall s_attack {r['recall_s_attack']:.3f}, IF {r['recall_if']:.3f}, "
                   f"union {r['recall_union']:.3f}; ")
        else:
            r["test_recall_all_attacks"] = float(det[~ben].mean()); msg = f"all-attack recall {r['test_recall_all_attacks']:.4f}; "
        out[tag] = r
        say(f"E2 {tag}: {msg}benign alerts {r['benign_alert_pct_s_attack']:.3f}% -> {r['benign_alert_pct_union']:.3f}% "
            f"(best_iter {r['best_iteration']}, fit {fit_s:.0f}s){'' if r['valid'] else '  INVALID (benign alerts > 1 %)'}")
    return out


def stage_e3(z, P, tau, core, names, n_each=500):
    import joblib
    import shap
    from scipy.stats import spearmanr
    pin({core})
    model = joblib.load(ROOT / "edge/models_v2/lgbm_classifier.pkl")
    bg = np.load(ROOT / "edge/models_v2/lgbm_background_100.npy").astype(np.float32)
    Xs, ys = z["X"][P["test"]], z["y"][P["test"]]
    s = onnx_scores(Xs)
    rng = np.random.default_rng(0)
    ia = rng.choice(np.where(s >= tau)[0], n_each, replace=False)
    ib = rng.choice(np.where(s < tau)[0], n_each, replace=False)
    idx = np.concatenate([ia, ib]); X = Xs[idx].astype(np.float32)
    pred = model.predict(X).astype(int)
    booster = model.booster_
    raw = booster.predict(X, raw_score=True)                   # (n, 6)
    n_cls = raw.shape[1]; nf = X.shape[1]

    exp_m0 = shap.TreeExplainer(model, data=bg)
    exp_m1 = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    def via_shap(explainer):
        def f(x, c):
            sv = explainer.shap_values(x)
            return np.asarray(sv[c][0] if isinstance(sv, list) else sv[0, :, c])
        return f

    def native(x, c):
        contrib = booster.predict(x, pred_contrib=True, num_threads=1).reshape(n_cls, nf + 1)
        return contrib[c, :nf]

    methods = {"M0_interventional_bg100": via_shap(exp_m0), "M1_tree_path_dependent": via_shap(exp_m1),
               "M2_lightgbm_pred_contrib": native}
    vals = {}; out = {"n_flows": int(len(idx)), "n_alerting": n_each, "n_non_alerting": n_each,
                      "explained_class": "argmax of the pkl model (as the API)", "core": core, "methods": {}}
    for name, f in methods.items():
        say(f"E3 {name}: warm-up")
        for i in range(1 if name.startswith("M0") else 5):
            f(X[i:i + 1], pred[i])
        lat, V, fails = [], np.zeros((len(X), nf)), 0
        for i in range(len(X)):
            t = time.perf_counter()
            try:
                V[i] = f(X[i:i + 1], pred[i])
            except Exception:  # noqa: BLE001
                fails += 1; V[i] = np.nan
            lat.append((time.perf_counter() - t) * 1000)
            if (i + 1) % (100 if name.startswith("M0") else 250) == 0:
                say(f"    {name}: {i + 1}/{len(X)} (mean {np.mean(lat):.2f} ms, failures {fails})")
        a = np.asarray(lat)
        out["methods"][name] = {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
                                "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
                                "max_ms": float(a.max()), "exceptions": fails}
        vals[name] = V
        say(f"E3 {name}: mean {a.mean():.2f} ms, p99 {np.percentile(a, 99):.2f} ms, exceptions {fails}")
    # additivity of the native method: contributions + bias == raw margin of the explained class
    full = booster.predict(X, pred_contrib=True).reshape(len(X), n_cls, nf + 1)
    out["methods"]["M2_lightgbm_pred_contrib"]["max_additivity_error"] = float(np.max(np.abs(
        full.sum(-1)[np.arange(len(X)), pred] - raw[np.arange(len(X)), pred])))
    ref = vals["M0_interventional_bg100"]
    for name in ("M1_tree_path_dependent", "M2_lightgbm_pred_contrib"):
        V = vals[name]; ok = ~np.isnan(ref).any(1) & ~np.isnan(V).any(1)
        jac, same5, top1, rho = [], 0, 0, []
        for i in np.where(ok)[0]:
            a5 = set(np.argsort(-np.abs(ref[i]))[:5]); b5 = set(np.argsort(-np.abs(V[i]))[:5])
            jac.append(len(a5 & b5) / len(a5 | b5)); same5 += a5 == b5
            top1 += int(np.argmax(np.abs(ref[i])) == np.argmax(np.abs(V[i])))
            r = spearmanr(ref[i], V[i]).correlation
            if not np.isnan(r):
                rho.append(r)
        n = int(ok.sum())
        out["methods"][name]["agreement_vs_M0"] = {"n": n, "top5_jaccard_mean": float(np.mean(jac)),
                                                   "top5_identical_pct": 100 * same5 / n, "top1_same_pct": 100 * top1 / n,
                                                   "spearman_median": float(np.median(rho))}
        say(f"E3 {name} vs M0: top-5 Jaccard {np.mean(jac):.3f}, identical top-5 {100 * same5 / n:.1f}%, "
            f"top-1 {100 * top1 / n:.1f}%")
    # M1 vs M2 should be the same algorithm (path-dependent TreeSHAP)
    d = np.nanmax(np.abs(vals["M1_tree_path_dependent"] - vals["M2_lightgbm_pred_contrib"]))
    out["M1_vs_M2_max_abs_diff"] = float(d)
    ok_methods = [m for m, v in out["methods"].items() if v["p99_ms"] <= 5.0 and v["exceptions"] == 0]
    out["selection_rule"] = "fastest method with p99 <= 5 ms and zero exceptions (fixed before the run)"
    out["selected"] = min(ok_methods, key=lambda m: out["methods"][m]["mean_ms"]) if ok_methods else None
    say(f"E3 selected: {out['selected']}")
    return out


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--stage", choices=["e1", "e2", "e3"], required=True)
    ap.add_argument("--core", type=int, default=2)
    a = ap.parse_args()
    ROOT = Path(a.repo).resolve(); sys.path.insert(0, str(ROOT))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{a.stage}.json"
    if dst.exists():
        say(f"{dst.name} exists - skipping"); return
    from edge.inference.decision import get_default_policy, model_binding
    from edge.training.label_mapper import CLASS_NAMES
    pol = get_default_policy()
    b = model_binding(ROOT / "edge/models_v2/lgbm_classifier.onnx", pol)
    assert b["match"] is True, b
    z, P = load_data(a.cache)
    fn = {"e1": stage_e1, "e2": stage_e2}.get(a.stage)
    res = fn(z, P, pol.alert_threshold, CLASS_NAMES) if fn else stage_e3(z, P, pol.alert_threshold, a.core, CLASS_NAMES)
    res["meta"] = {"repo_commit": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(), "policy": pol.describe(),
                   "elapsed_s": round(time.time() - _T0, 1), "cpu_count": os.cpu_count()}
    dst.write_text(json.dumps(res, indent=1, default=str))
    say(f"written {dst}")


if __name__ == "__main__":
    import logging
    import warnings
    logging.disable(logging.WARNING); warnings.filterwarnings("ignore")
    main()
