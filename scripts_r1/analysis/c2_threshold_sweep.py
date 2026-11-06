"""
C2 — Threshold sweep for s_attack = 1 - p_BENIGN on VALIDATION only.

Usage:
  python c2_threshold_sweep.py val            # sweep on validation, write tables
  python c2_threshold_sweep.py test <tau>     # evaluate ONE chosen tau on test
No source edits, no retraining. Outputs under revision-audit/reports/threshold/.
Encryption cost uses the real-TenSEAL microbenchmark from C1 (6.5677 ms, 441,400 B per vector).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/path/to/SAINT/realtime-ids-healthcareBeta")
AUD = Path("/path/to/SAINT/revision-audit")
OUT = AUD / "reports" / "threshold"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import onnxruntime as ort  # noqa: E402
import edge.training.dataset_loader as dl  # noqa: E402
from edge.training.feature_mapper import map_cicids_to_features  # noqa: E402
from edge.training.label_mapper import CLASS_NAMES, map_cicids_labels  # noqa: E402

c1 = json.loads((AUD / "reports" / "policy_decision" / "policy_comparison_results.json").read_text())
ENC_MS = c1["latency_microbench"]["ckks_encrypt_hmac_28field"]["p50_ms"]
CT_BYTES = c1["latency_microbench"]["ckks_ciphertext_b64_bytes"]
V1_RATE = 4269  # flows/s, V1 Table 6 (model-only throughput), used only to scale workload


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(float(c - h), 6), round(float(c + h), 6))


def load_split(which):
    dl.CICIDS2017_DIR = ROOT / "data" / "cicids2017"
    df = dl.load_cicids2017()
    X = map_cicids_to_features(df); y = map_cicids_labels(df["Label"])
    z = np.load(AUD / "work" / "v1_split_reproduction.npz")
    assert np.array_equal(y, z["y"])
    idx = z["i_val"] if which == "val" else z["i_test"]
    sess = ort.InferenceSession(str(ROOT / "edge/models/lgbm_classifier.onnx"), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    Xs = X[idx].astype(np.float32); P = np.zeros((len(idx), 6))
    for s in range(0, len(idx), 50000):
        for r, d in enumerate(sess.run(None, {inp: Xs[s:s + 50000]})[1]):
            for k, v in d.items():
                P[s + r, int(k)] = v
    return y[idx], P


def row(y, P, tau):
    s = 1 - P[:, 0]; al = s >= tau; atk = y != 0; ben = ~atk; n = len(y)
    tp, fp = int((al & atk).sum()), int((al & ben).sum()); fn, tn = int((~al & atk).sum()), int((~al & ben).sum())
    rec = tp / (tp + fn); prec = tp / (tp + fp) if tp + fp else 0.0
    am = P.argmax(1); cfp = int(((am != 0) & ben).sum())
    r = {"tau": tau, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
         "recall": rec, "recall_CI95": wilson(tp, tp + fn), "precision": prec,
         "F1_binary": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
         "binary_FPR_eq_benign_alert_pct": 100 * fp / (fp + tn), "benign_alert_CI95_pct": [100 * v for v in wilson(fp, fp + tn)],
         "argmax_classifier_FPR_pct_fixed": 100 * cfp / int(ben.sum()),
         "all_flow_alert_pct": 100 * (tp + fp) / n, "encrypt_selected": tp + fp, "encrypt_pct": 100 * (tp + fp) / n,
         "false_alert_share_pct": 100 * fp / (tp + fp) if tp + fp else None,
         "missed_attacks": fn, "false_alarms_per_10k_benign": 1e4 * fp / (fp + tn),
         "ckks_cpu_s_per_1k_flows": (tp + fp) / n * 1000 * ENC_MS / 1000,
         "ckks_cores_at_4269fps": (tp + fp) / n * V1_RATE * ENC_MS / 1000,
         "ciphertext_MB_per_s_at_4269fps": (tp + fp) / n * V1_RATE * CT_BYTES / 1e6}
    for k in range(1, 6):
        m = y == k
        r[f"{CLASS_NAMES[k]}_alerted"] = int(al[m].sum()); r[f"{CLASS_NAMES[k]}_missed"] = int((~al[m]).sum())
        r[f"{CLASS_NAMES[k]}_recall"] = float(al[m].mean())
    return r


mode = sys.argv[1]
if mode == "val":
    y, P = load_split("val")
    taus = [1e-4, 1e-3, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    rows = [row(y, P, t) for t in taus]
    pd.DataFrame(rows).to_csv(OUT / "val_threshold_table.csv", index=False)
    # score distribution in the decision region (explains why the curve is flat)
    s = 1 - P[:, 0]; atk = y != 0
    edges = [0, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0000001]
    dist = pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:],
                         "benign": [int(((s >= a) & (s < b) & ~atk).sum()) for a, b in zip(edges[:-1], edges[1:])],
                         "attack": [int(((s >= a) & (s < b) & atk).sum()) for a, b in zip(edges[:-1], edges[1:])]})
    dist.to_csv(OUT / "val_score_distribution.csv", index=False)
    # attack-type naming (threshold-independent): argmax among all classes for true attacks
    cm = pd.crosstab(pd.Series(y).map(dict(enumerate(CLASS_NAMES))), pd.Series(P.argmax(1)).map(dict(enumerate(CLASS_NAMES))))
    cm.to_csv(OUT / "val_confusion_matrix_6class.csv")
    json.dump({"n": int(len(y)), "benign": int((~atk).sum()), "attack": int(atk.sum()),
               "per_class": {CLASS_NAMES[k]: int((y == k).sum()) for k in range(6)},
               "enc_ms_per_vector": ENC_MS, "ciphertext_bytes": CT_BYTES}, open(OUT / "val_meta.json", "w"), indent=2)
    print(pd.DataFrame(rows)[["tau", "TP", "FP", "FN", "recall", "precision", "F1_binary", "binary_FPR_eq_benign_alert_pct",
                              "all_flow_alert_pct", "ckks_cores_at_4269fps"]].to_string(index=False))
    print(dist.to_string(index=False))
elif mode == "test":
    tau = float(sys.argv[2]); y, P = load_split("test")
    r = row(y, P, tau)
    json.dump(r, open(OUT / f"test_eval_tau_{tau}.json", "w"), indent=2, default=str)
    cm = pd.crosstab(pd.Series(y).map(dict(enumerate(CLASS_NAMES))), pd.Series(P.argmax(1)).map(dict(enumerate(CLASS_NAMES))))
    cm.to_csv(OUT / "test_confusion_matrix_6class.csv")
    print(json.dumps(r, indent=1, default=str))
