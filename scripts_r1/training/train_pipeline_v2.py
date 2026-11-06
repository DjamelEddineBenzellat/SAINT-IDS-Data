"""
Training pipeline v2 — deduplicated, leakage-checked protocol (Reviewer 1, Comment 8).

The V1 pipeline (train_pipeline.py) is left unchanged so the submitted results
stay reproducible. v2 reuses its building blocks and changes only the protocol:

  load → map 33 features → DEDUPLICATE (conflicting labels dropped) →
  4-way stratified split (train / val / tune / test) → inf replacement and
  balancing fitted on TRAIN only → LightGBM (baseline, seeded Optuna on val,
  refit on train+val) → Isolation Forest (BENIGN train rows) → evaluation on
  tune and test → leakage check (0 shared vectors) → artifacts + report.

The tune partition is never used for fitting; it exists to select the alert
threshold of policy C′. Test is evaluated once.

Usage:
  python -m edge.training.train_pipeline_v2 --data-dir data/cicids2017 [--pilot --out-dir DIR]
  python -m edge.training.train_pipeline_v2 --data-dir data/cicids2017            # full run → edge/models_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from edge.training.dedup import cross_partition_overlap, deduplicate
from edge.training.feature_mapper import map_cicids_to_features
from edge.training.label_mapper import CLASS_NAMES, NUM_CLASSES, map_cicids_labels
from edge.training.train_pipeline import (
    balance_classes,
    detect_gpu_device,
    train_isolation_forest,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "v2_dedup.json"
_INF_COLUMNS = {"Flow Packets/s": 3, "Flow Bytes/s": 23}   # raw column → feature index


# =============================================================================
# Protocol steps (pure functions; unit-tested)
# =============================================================================

def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    cfg = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    fr = cfg["split"]
    total = fr["train"] + fr["val"] + fr["tune"] + fr["test"]
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    return cfg


def split_four_way(y: np.ndarray, fractions: Dict[str, float], random_state: int = 42) -> Dict[str, np.ndarray]:
    """Stratified train/val/tune/test indices (positions into y)."""
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=fractions["test"], stratify=y, random_state=random_state)
    rem = 1.0 - fractions["test"]
    rest2, tune = train_test_split(rest, test_size=fractions["tune"] / rem, stratify=y[rest],
                                   random_state=random_state)
    rem2 = rem - fractions["tune"]
    train, val = train_test_split(rest2, test_size=fractions["val"] / rem2, stratify=y[rest2],
                                  random_state=random_state)
    return {"train": np.sort(train), "val": np.sort(val), "tune": np.sort(tune), "test": np.sort(test)}


def replace_inf_train_only(X: np.ndarray, inf_masks: Dict[int, np.ndarray], train_idx: np.ndarray) -> Dict:
    """
    feature_mapper replaces ±inf by the full-data column max. Re-set those cells
    to the max of finite *training* values so no statistic comes from val/tune/test.
    Modifies X in place; returns what was done.
    """
    out = {}
    in_train = np.zeros(len(X), dtype=bool)
    in_train[train_idx] = True
    for col, mask in inf_masks.items():
        finite_train = in_train & ~mask
        value = float(X[finite_train, col].max()) if finite_train.any() else 0.0
        before = float(X[mask, col][0]) if mask.any() else None
        X[mask, col] = value
        out[str(col)] = {"inf_cells": int(mask.sum()), "full_data_value": before, "train_only_value": value}
    return out


def evaluate_classifier(model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Same quantities as V1 (argmax classifier), plus supports and counts."""
    labels = list(range(NUM_CLASSES))
    proba = model.predict_proba(X)
    pred = model.classes_[np.argmax(proba, axis=1)]
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0.0)
    ben = y == 0
    fp = int(np.sum((pred != 0) & ben)); tn = int(np.sum((pred == 0) & ben))
    return {
        "n": int(len(y)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", labels=labels, zero_division=0.0)),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "fp": fp, "tn": tn,
        "precision_per_class": {CLASS_NAMES[i]: float(p[i]) for i in labels},
        "recall_per_class": {CLASS_NAMES[i]: float(r[i]) for i in labels},
        "f1_per_class": {CLASS_NAMES[i]: float(f[i]) for i in labels},
        "support_per_class": {CLASS_NAMES[i]: int(s[i]) for i in labels},
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def gates_report(metrics: Dict[str, Any]) -> Dict[str, bool]:
    """V1 validation gates, recorded (not raised) so v2 results are always reported."""
    g = {"f1_weighted_gt_0_95": metrics["f1_weighted"] > 0.95,
         "fpr_lt_0_02": metrics["false_positive_rate"] < 0.02}
    for c in ["PORT_SCAN", "BRUTE_FORCE", "DOS"]:
        g[f"{c.lower()}_recall_gt_0_90"] = metrics["recall_per_class"][c] > 0.90
    return g


def class_counts(y: np.ndarray) -> Dict[str, int]:
    return {CLASS_NAMES[k]: int(np.sum(y == k)) for k in range(NUM_CLASSES)}


# =============================================================================
# LightGBM (V1 search space; seeded sampler; deterministic CPU training)
# =============================================================================

def _base_params(device: str, gpu_params: dict, n_jobs: int = -1) -> dict:
    return {"objective": "multiclass", "class_weight": "balanced", "random_state": 42,
            "n_jobs": n_jobs if device == "cpu" else 1, "verbose": -1,
            "deterministic": True, "force_col_wise": True, **gpu_params}


def fit_lightgbm(Xtr, ytr, Xva, yva, cfg: Dict, n_trials: int, record_curves: bool = True,
                 n_jobs: int = -1, storage: Optional[str] = None, device: str = "cpu") -> Dict[str, Any]:
    """
    storage: optional Optuna RDB URL (e.g. sqlite:///…/optuna_v2.db). Completed
    trials are persisted, and a rerun resumes until `n_trials` trials are complete.
    """
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # "cpu" (default) keeps the approved protocol: GPU mode changes the search space and determinism.
    device, gpu_params = ("cpu", {}) if device == "cpu" else detect_gpu_device()
    base = _base_params(device, gpu_params, n_jobs)
    es = cfg["lightgbm"]["optuna"]["early_stopping_rounds"]
    out: Dict[str, Any] = {"device": device, "timings_s": {}}

    t = time.time()
    b = cfg["lightgbm"]["baseline"]
    bm = lgb.LGBMClassifier(**base, **b)
    bm.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
    out["baseline_val_f1_weighted"] = float(f1_score(yva, bm.predict(Xva), average="weighted"))
    out["timings_s"]["baseline"] = round(time.time() - t, 1)
    logger.info("Baseline val F1w=%.6f (%.0fs)", out["baseline_val_f1_weighted"], out["timings_s"]["baseline"])

    max_leaves = 64 if device == "gpu" else 150
    min_child = 20 if device == "gpu" else 5

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 20, max_leaves),
            "min_child_samples": trial.suggest_int("min_child_samples", min_child, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        t0 = time.time()
        clf = lgb.LGBMClassifier(**base, **params)
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
        val = float(f1_score(yva, clf.predict(Xva), average="weighted"))
        trial.set_user_attr("fit_seconds", round(time.time() - t0, 1))
        trial.set_user_attr("best_iteration", int(clf.best_iteration_ or params["n_estimators"]))
        logger.info("trial %d: val F1w=%.6f (%.0fs)", trial.number, val, time.time() - t0)
        return val

    t = time.time()
    seed = cfg["lightgbm"]["optuna"]["sampler_seed"]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                storage=storage, study_name=cfg["protocol_id"], load_if_exists=True)
    done = sum(tr.state == optuna.trial.TrialState.COMPLETE for tr in study.trials)
    remaining = max(0, n_trials - done)
    out["optuna_resume"] = {"storage": storage, "completed_before_this_session": done,
                            "run_this_session": remaining}
    if done:
        logger.info("Resuming Optuna study: %d trials already complete, %d to run", done, remaining)
    study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    out["timings_s"]["optuna"] = round(time.time() - t, 1)
    out["optuna_best_params"] = study.best_params
    out["optuna_best_val_f1"] = float(study.best_value)
    out["optuna_trials"] = [{"number": tr.number, "value": tr.value, "params": tr.params, **tr.user_attrs}
                            for tr in study.trials]

    if record_curves:   # Figure 5: early-stopped fit of the best configuration (same run, same split)
        t = time.time()
        ev: Dict = {}
        cm = lgb.LGBMClassifier(**base, **study.best_params)
        cm.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], eval_names=["train", "validation"],
               callbacks=[lgb.early_stopping(es, verbose=False), lgb.record_evaluation(ev), lgb.log_evaluation(0)])
        n_it = len(ev["validation"]["multi_logloss"])
        rng = np.random.RandomState(42)
        sub = rng.choice(len(ytr), size=min(200_000, len(ytr)), replace=False)
        its = sorted(set(np.linspace(1, n_it, num=min(25, n_it)).astype(int).tolist()))
        f1_tr, f1_va = [], []
        for it in its:
            f1_tr.append(float(f1_score(ytr[sub], cm.predict(Xtr[sub], num_iteration=it), average="weighted")))
            f1_va.append(float(f1_score(yva, cm.predict(Xva, num_iteration=it), average="weighted")))
        out["curves"] = {"train_logloss": ev["train"]["multi_logloss"], "val_logloss": ev["validation"]["multi_logloss"],
                         "best_iteration": int(cm.best_iteration_ or n_it), "f1_iterations": its,
                         "train_f1_weighted_subsample200k": f1_tr, "val_f1_weighted": f1_va,
                         "note": "early-stopped fit of the Optuna-best configuration on train, monitored on val"}
        out["timings_s"]["curves"] = round(time.time() - t, 1)

    t = time.time()
    Xtv = np.concatenate([Xtr, Xva]); ytv = np.concatenate([ytr, yva])
    final_params = refit_params(study.best_params, study.best_trial.user_attrs)
    final = lgb.LGBMClassifier(**base, **final_params)
    final.fit(Xtv, ytv)
    out["timings_s"]["final_refit"] = round(time.time() - t, 1)
    # Sanity check: val is part of the refit data, so the refit must not score
    # clearly below the trial's early-stopped val F1 (it diverged in the pilot).
    refit_val = float(f1_score(yva, final.predict(Xva), average="weighted"))
    out["refit_check"] = {"n_estimators_used": final_params["n_estimators"],
                          "n_estimators_searched": study.best_params["n_estimators"],
                          "best_trial_val_f1": float(study.best_value), "refit_val_f1": refit_val,
                          "ok": refit_val >= float(study.best_value) - 0.005}
    if not out["refit_check"]["ok"]:
        logger.warning("Refit check FAILED: %s", out["refit_check"])
    out["model"] = final
    out["lgbm_params"] = {**base, **final_params}
    return out


def refit_params(best_params: Dict[str, Any], best_trial_attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final-refit parameters: the selected trial's hyper-parameters with
    n_estimators set to that trial's early-stopping best iteration.

    V1 refit with the searched n_estimators and no early stopping; the v2 pilot
    showed this can run far past the optimum (best iteration 42 of 393 at lr 0.25)
    and wreck the model (test F1w 0.60).
    """
    p = dict(best_params)
    best_it = int(best_trial_attrs.get("best_iteration") or p["n_estimators"])
    p["n_estimators"] = max(1, min(best_it, int(p["n_estimators"])))
    return p


def shap_background(X: np.ndarray, y: np.ndarray, n: int = 100) -> np.ndarray:
    """V1 recipe: up to n // NUM_CLASSES samples per class from the training data."""
    rng = np.random.RandomState(42)
    per = max(1, n // NUM_CLASSES)
    parts = [X[rng.choice(np.where(y == c)[0], size=min(per, int((y == c).sum())), replace=False)]
             for c in range(NUM_CLASSES) if (y == c).any()]
    return np.concatenate(parts)[:n]


# =============================================================================
# Provenance helpers
# =============================================================================

def _git_state() -> Dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=_PROJECT_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return {"commit": head, "tracked_files_modified": bool(dirty)}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> Dict[str, str]:
    import importlib
    v = {"python": platform.python_version()}
    for m in ["numpy", "pandas", "sklearn", "lightgbm", "optuna", "imblearn", "onnxruntime", "skl2onnx", "onnxmltools"]:
        try:
            v[m] = importlib.import_module(m).__version__
        except Exception:  # noqa: BLE001
            v[m] = "unavailable"
    return v


# =============================================================================
# Pipeline
# =============================================================================

def prepare_data(data_dir: Path | str | None, cfg: Dict) -> Dict[str, Any]:
    """Load → map → dedup → split → train-only inf replacement → train-only balancing."""
    from edge.training import dataset_loader as dl

    directory = Path(data_dir or os.environ.get("SAINT_CICIDS_DIR") or dl.CICIDS2017_DIR)
    df = dl.load_cicids2017(directory)
    inf_raw = {col: np.isinf(df[name].to_numpy(dtype=float)) for name, col in _INF_COLUMNS.items()}
    X = map_cicids_to_features(df)
    y = map_cicids_labels(df["Label"])
    n_raw = len(y)
    del df
    info: Dict[str, Any] = {"rows_raw": n_raw, "class_counts_raw": class_counts(y)}
    info["dataset_dir"] = str(directory)
    info["dataset_sha256"] = {f: _file_sha256(directory / f) for f in dl.CICIDS2017_FILES}

    keep, dedup_report = deduplicate(X, y, cfg["dedup"]["conflict_policy"])
    X, y = X[keep], y[keep]
    inf_masks = {c: m[keep] for c, m in inf_raw.items()}
    info["dedup"] = dedup_report

    parts = split_four_way(y, cfg["split"], cfg["split"]["random_state"])
    info["split_counts"] = {k: class_counts(y[v]) for k, v in parts.items()}
    info["split_sizes"] = {k: int(len(v)) for k, v in parts.items()}
    info["inf_replacement"] = replace_inf_train_only(X, inf_masks, parts["train"])

    trval = np.concatenate([parts["train"], parts["val"]])
    info["leakage_check"] = {
        "test_rows_with_vector_in_train_or_val": cross_partition_overlap(X, trval, parts["test"]),
        "tune_rows_with_vector_in_train_or_val": cross_partition_overlap(X, trval, parts["tune"]),
        "test_rows_with_vector_in_tune": cross_partition_overlap(X, parts["tune"], parts["test"]),
    }
    if any(info["leakage_check"].values()):
        raise RuntimeError(f"leakage check failed: {info['leakage_check']}")

    b = cfg["balancing"]
    Xtr, ytr = balance_classes(X[parts["train"]], y[parts["train"]], b["max_benign_multiplier"],
                               b["smote_threshold"], b["random_state"])
    info["balancing"] = {"applied_to": "train", "rows_before": int(len(parts["train"])), "rows_after": int(len(ytr)),
                         "class_counts_after": class_counts(ytr),
                         "is_noop": bool(len(ytr) == len(parts["train"]))}
    return {"X": X, "y": y, "parts": parts, "Xtr": Xtr, "ytr": ytr, "info": info}


def _prepare_cached(data_dir, cfg: Dict, cache_dir: Optional[Path]) -> Dict[str, Any]:
    """prepare_data(), cached in cache_dir (keyed by the protocol config) so a crash or rerun skips it."""
    if cache_dir is None:
        return prepare_data(data_dir, cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    npz, meta = cache_dir / f"prepared_{key}.npz", cache_dir / f"prepared_{key}.json"
    if npz.exists() and meta.exists():
        z = np.load(npz)
        logger.info("Loaded prepared data from cache %s", npz)
        return {"X": z["X"], "y": z["y"], "Xtr": z["Xtr"], "ytr": z["ytr"],
                "parts": {k: z[f"part_{k}"] for k in ("train", "val", "tune", "test")},
                "info": {**json.loads(meta.read_text()), "loaded_from_cache": str(npz)}}
    d = prepare_data(data_dir, cfg)
    tmp = cache_dir / f"prepared_{key}.tmp.npz"
    np.savez(tmp, X=d["X"], y=d["y"], Xtr=d["Xtr"], ytr=d["ytr"], **{f"part_{k}": v for k, v in d["parts"].items()})
    tmp.replace(npz)                                  # atomic: a crash never leaves a half-written cache
    meta.write_text(json.dumps(d["info"], indent=1, default=str))
    return d


def run_v2(data_dir=None, config_path=None, out_dir=None, n_trials: Optional[int] = None,
           pilot: bool = False, control: bool = True, n_jobs: int = -1,
           cache_dir: Path | str | None = None, stage: str = "all", device: str = "cpu") -> Dict[str, Any]:
    """
    stage="prepare": only build (and cache) the deduplicated split, then stop.
    cache_dir: holds the prepared-data cache and the Optuna SQLite study so an
               interrupted run resumes. Keep it outside the repository.
    """
    import joblib

    cfg = load_config(config_path)
    cache = Path(cache_dir) if cache_dir else None
    out = Path(out_dir or _PROJECT_ROOT / cfg["output_dir"])
    if not pilot and (out / "lgbm_classifier.pkl").exists():
        raise FileExistsError(f"{out} already contains a model; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    trials = 1 if pilot else (n_trials or cfg["lightgbm"]["optuna"]["n_trials"])

    d = _prepare_cached(data_dir, cfg, cache)
    X, y, P = d["X"], d["y"], d["parts"]
    logger.info("Prepared: %s", d["info"]["split_sizes"])
    if stage == "prepare":
        return {"stage": "prepare", "data": d["info"], "elapsed_s": round(time.time() - t_start, 1)}

    storage = None
    if cache is not None and not pilot:
        storage = f"sqlite:///{(cache / 'optuna_v2.db').resolve()}"
    lg = fit_lightgbm(d["Xtr"], d["ytr"], X[P["val"]], y[P["val"]], cfg, trials, record_curves=not pilot,
                      n_jobs=n_jobs, storage=storage, device=device)
    model = lg.pop("model")

    report: Dict[str, Any] = {
        "protocol": cfg, "mode": "pilot" if pilot else "full",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(), "versions": _versions(), "cpu_count": os.cpu_count(), "platform": platform.platform(),
        "data": d["info"], "lightgbm": {k: v for k, v in lg.items() if k != "curves"},
    }
    report["evaluation"] = {"tune": evaluate_classifier(model, X[P["tune"]], y[P["tune"]]),
                            "test": evaluate_classifier(model, X[P["test"]], y[P["test"]])}
    report["gates_on_test"] = gates_report(report["evaluation"]["test"])

    if pilot:
        report["elapsed_s"] = round(time.time() - t_start, 1)
        (out / "pilot_report.json").write_text(json.dumps(report, indent=2, default=str))
        return report

    # Isolation Forest (logged signal; BENIGN rows of the balanced train set)
    if_model, if_metrics, if_bg = train_isolation_forest(d["Xtr"][d["ytr"] == 0], X[P["test"]], y[P["test"]])
    report["isolation_forest_metrics_test"] = if_metrics

    # Control: V1 hyperparameters on the same v2 split (separates dedup from re-tuning)
    if control:
        import lightgbm as lgb
        v1 = json.loads((_PROJECT_ROOT / "edge/models/training_report.json").read_text())["optuna_best_params"]
        dev, gpu = ("cpu", {}) if device == "cpu" else detect_gpu_device()
        t = time.time()
        cm = lgb.LGBMClassifier(**_base_params(dev, gpu, n_jobs), **v1)
        cm.fit(np.concatenate([d["Xtr"], X[P["val"]]]), np.concatenate([d["ytr"], y[P["val"]]]))
        report["control_v1_params"] = {"params": v1, "fit_seconds": round(time.time() - t, 1),
                                       "tune": evaluate_classifier(cm, X[P["tune"]], y[P["tune"]]),
                                       "test": evaluate_classifier(cm, X[P["test"]], y[P["test"]])}

    trval_y = np.concatenate([d["ytr"], y[P["val"]]])
    trval_X = np.concatenate([d["Xtr"], X[P["val"]]])
    joblib.dump(model, out / "lgbm_classifier.pkl")
    joblib.dump(if_model, out / "isolation_forest.pkl")
    np.save(out / "lgbm_background_100.npy", shap_background(trval_X, trval_y))
    np.save(out / "if_background_100.npy", if_bg)
    np.savez_compressed(out / "split_indices_v2.npz", **P)
    (out / "training_curves_v2.json").write_text(json.dumps(lg.get("curves", {}), indent=1))
    report["lgbm_metrics"] = report["evaluation"]["test"]          # V1-compatible key for figure scripts
    report["optuna_best_params"] = lg["optuna_best_params"]
    report["models_saved"] = {k: str((out / f).relative_to(_PROJECT_ROOT)) for k, f in [
        ("lgbm_classifier", "lgbm_classifier.pkl"), ("isolation_forest", "isolation_forest.pkl"),
        ("lgbm_background", "lgbm_background_100.npy"), ("if_background", "if_background_100.npy"),
        ("split_indices", "split_indices_v2.npz"), ("training_curves", "training_curves_v2.json")]}
    (out / "training_report.json").write_text(json.dumps(report, indent=2, default=str))

    # ONNX export (FP32 + INT8) into the same directory; exporter appends `onnx_export`
    from edge.training.onnx_exporter import run_export_pipeline
    exp = run_export_pipeline(models_dir=out)
    report = json.loads((out / "training_report.json").read_text())
    report["onnx_export_gates"] = exp["gates"]
    report["onnx_fidelity_real_test"] = _onnx_fidelity(out, model, X[P["test"]])
    report["elapsed_s"] = round(time.time() - t_start, 1)
    (out / "training_report.json").write_text(json.dumps(report, indent=2, default=str))
    logger.info("v2 artifacts written to %s (%.0fs)", out, report["elapsed_s"])
    return report


def _onnx_fidelity(out: Path, model, X: np.ndarray) -> Dict[str, Any]:
    """pkl vs ONNX on REAL test vectors (V1 only checked random Gaussian vectors)."""
    import onnxruntime as ort
    res = {}
    p_pkl = model.predict_proba(X)
    for name in ["lgbm_classifier.onnx", "lgbm_classifier_int8.onnx"]:
        sess = ort.InferenceSession(str(out / name), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        P = np.zeros_like(p_pkl)
        for s in range(0, len(X), 50000):
            for r, d in enumerate(sess.run(None, {inp: X[s:s + 50000].astype(np.float32)})[1]):
                for k, v in d.items():
                    P[s + r, int(k)] = v
        res[name] = {"n": int(len(X)), "argmax_agreement": float(np.mean(P.argmax(1) == p_pkl.argmax(1))),
                     "max_abs_prob_diff": float(np.max(np.abs(P - p_pkl))),
                     "mean_abs_prob_diff": float(np.mean(np.abs(P - p_pkl)))}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir"); ap.add_argument("--config"); ap.add_argument("--out-dir")
    ap.add_argument("--trials", type=int); ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1, help="LightGBM threads (default: all cores)")
    ap.add_argument("--cache-dir", help="prepared-data cache + resumable Optuna study (outside the repo)")
    ap.add_argument("--stage", choices=["all", "prepare"], default="all")
    ap.add_argument("--device", choices=["cpu", "auto"], default="cpu",
                    help="cpu (default, approved protocol) or auto (use GPU if nvidia-smi is found)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    r = run_v2(a.data_dir, a.config, a.out_dir, a.trials, a.pilot, not a.no_control,
               a.n_jobs, a.cache_dir, a.stage, a.device)
    if a.stage == "prepare":
        print(json.dumps({k: r["data"][k] for k in ("split_sizes", "leakage_check", "balancing")}, indent=1))
        print(f"prepare stage done in {r['elapsed_s']}s")
        sys.exit(0)
    t = r["evaluation"]["test"]
    print(f"\n{'PILOT' if a.pilot else 'v2'} done in {r['elapsed_s']}s — test F1w={t['f1_weighted']:.6f} FPR={t['false_positive_rate']:.6f}")
