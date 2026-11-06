"""
C4 — Comment 8: V1 vs v2 comparison (read-only; no training).

Inputs : edge/models (V1), edge/models_v2 (v2), work/v1_split_reproduction.npz,
         work/v2_cache/prepared_5ebccf745c0b4863.npz
Output : reports/comment8/comment8_v1_vs_v2.json

- Re-predicts the V1 test set with the V1 model (must reproduce Table 3) and the
  v2 tune/test sets with the v2 model (must reproduce training_report.json).
- V1 test subsets matching v2 test conditions: vector unseen in V1 train∪val,
  vector not label-conflicting (absent from the v2 deduplicated pool), first copy only.
- 95 % CIs: exact multinomial bootstrap over confusion-matrix cells (B=2000, seed 0)
  for weighted F1 / FPR; Wilson intervals for FPR and per-class recall.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score

AUD = Path(__file__).resolve().parents[1]
REPO = AUD.parent / "realtime-ids-healthcareBeta"
sys.path.insert(0, str(REPO))
from edge.training.dedup import row_hashes  # noqa: E402
from edge.training.label_mapper import CLASS_NAMES  # noqa: E402

K = len(CLASS_NAMES)
RNG = np.random.default_rng(0)
B = 2000


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def f1w_from_cm(cm):
    tp = np.diag(cm).astype(float); sup = cm.sum(1); pred = cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred > 0, tp / pred, 0); rec = np.where(sup > 0, tp / sup, 0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
    return float((f1 * sup).sum() / sup.sum())


def fpr_from_cm(cm):
    return float(1 - cm[0, 0] / cm[0].sum())


def summarize(y, p, name):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    n = cm.sum(); probs = (cm / n).ravel()
    boots = RNG.multinomial(n, probs, size=B).reshape(B, K, K)
    f1b = np.array([f1w_from_cm(c) for c in boots]); fprb = np.array([fpr_from_cm(c) for c in boots])
    f1 = f1w_from_cm(cm)
    assert abs(f1 - f1_score(y, p, average="weighted")) < 1e-9
    fp = int(cm[0].sum() - cm[0, 0])
    per = {}
    for k, c in enumerate(CLASS_NAMES):
        s = int(cm[k].sum())
        if s == 0:
            continue
        tp = int(cm[k, k]); pr = int(cm[:, k].sum())
        prec = tp / pr if pr else 0.0; rec = tp / s
        per[c] = {"support": s, "recall": rec, "recall_wilson95": wilson(tp, s), "precision": prec,
                  "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    return {"name": name, "n": int(n), "f1_weighted": f1,
            "f1_weighted_boot95": [float(np.percentile(f1b, 2.5)), float(np.percentile(f1b, 97.5))],
            "fpr": fpr_from_cm(cm), "fp": fp, "benign": int(cm[0].sum()),
            "fpr_wilson95": wilson(fp, int(cm[0].sum())),
            "fpr_boot95": [float(np.percentile(fprb, 2.5)), float(np.percentile(fprb, 97.5))],
            "per_class": per, "confusion_matrix": cm.tolist()}


def predict(model, X):
    model.set_params(n_jobs=2)
    return model.predict(X).astype(np.int64)


def main():
    out = {}
    # ── V1 ────────────────────────────────────────────────────────────────
    z1 = np.load(AUD / "work/v1_split_reproduction.npz")
    X1 = z1["X_test"].astype(np.float32); y1 = z1["y"][z1["i_test"]].astype(np.int64)
    seen = z1["seen_vec_test"].astype(bool)
    p1 = predict(joblib.load(REPO / "edge/models/lgbm_classifier.pkl"), X1)
    v1_rep = json.loads((REPO / "edge/models/training_report.json").read_text())["lgbm_metrics"]
    full = summarize(y1, p1, "V1 model / V1 test (Table 3)")
    full["reproduces_table3_confusion"] = full["confusion_matrix"] == v1_rep["confusion_matrix"]
    out["v1_full"] = full

    z2 = np.load(AUD / "work/v2_cache/prepared_5ebccf745c0b4863.npz")
    pool = np.unique(row_hashes(z2["X"].astype(np.float32)))      # non-conflicting unique vectors
    h1 = row_hashes(X1)
    conflict = ~np.isin(h1, pool)
    first = np.zeros(len(h1), bool); first[np.unique(h1, return_index=True)[1]] = True
    out["v1_subset_counts"] = {"test": int(len(y1)), "seen_in_train_or_val": int(seen.sum()),
                               "label_conflicting_vector": int(conflict.sum()),
                               "conflicting_by_class": {CLASS_NAMES[k]: int((conflict & (y1 == k)).sum()) for k in range(K)},
                               "duplicate_copies_within_test": int((~first).sum())}
    m = ~seen
    out["v1_unseen"] = summarize(y1[m], p1[m], "V1 model / V1 test, vectors unseen in train∪val")
    m = ~seen & ~conflict & first
    out["v1_v2like"] = summarize(y1[m], p1[m], "V1 model / V1 test, unseen + non-conflicting + deduplicated")
    m = ~conflict
    out["v1_nonconflict"] = summarize(y1[m], p1[m], "V1 model / V1 test, non-conflicting vectors only")

    # ── v2 ────────────────────────────────────────────────────────────────
    s = np.load(REPO / "edge/models_v2/split_indices_v2.npz")
    m2 = joblib.load(REPO / "edge/models_v2/lgbm_classifier.pkl")
    rep2 = json.loads((REPO / "edge/models_v2/training_report.json").read_text())
    for part in ("tune", "test"):
        X, y = z2["X"][s[part]], z2["y"][s[part]].astype(np.int64)
        r = summarize(y, predict(m2, X), f"v2 model / v2 {part}")
        r["reproduces_training_report"] = r["confusion_matrix"] == rep2["evaluation"][part]["confusion_matrix"]
        out[f"v2_{part}"] = r
    c = rep2["control_v1_params"]
    for part in ("tune", "test"):
        # the control model is not saved; report its stored metrics (no confusion-matrix bootstrap)
        out[f"control_{part}"] = {"name": f"V1 hyperparameters refit on v2 train∪val / v2 {part}",
                                  "f1_weighted": c[part]["f1_weighted"], "fpr": c[part]["false_positive_rate"],
                                  "fp": c[part]["fp"], "benign": c[part]["fp"] + c[part]["tn"],
                                  "fpr_wilson95": wilson(c[part]["fp"], c[part]["fp"] + c[part]["tn"]),
                                  "per_class": {k: {"support": c[part]["support_per_class"][k],
                                                    "recall": c[part]["recall_per_class"][k],
                                                    "precision": c[part]["precision_per_class"][k],
                                                    "f1": c[part]["f1_per_class"][k]}
                                                for k in CLASS_NAMES if c[part]["support_per_class"][k]}}
    dst = AUD / "reports/comment8/comment8_v1_vs_v2.json"
    dst.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if "f1_weighted" in v:
            print(f"{k:16s} n={v.get('n', '-'):>7} F1w={v['f1_weighted']:.6f} {v.get('f1_weighted_boot95', '')} "
                  f"FPR={v['fpr']*100:.4f}% {[round(x*100, 4) for x in v['fpr_wilson95']]}")
    print(json.dumps(out["v1_subset_counts"]), "repro:", out["v1_full"]["reproduces_table3_confusion"],
          out["v2_tune"]["reproduces_training_report"], out["v2_test"]["reproduces_training_report"])


if __name__ == "__main__":
    main()
