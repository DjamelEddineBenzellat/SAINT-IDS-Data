"""
C5 — Re-derive τ for s_attack = 1 − p_BENIGN on the v2 TUNE split (held out from the final model).

Usage:
  python c5_threshold_sweep_v2.py tune            # sweep on tune only, write tables
  python c5_threshold_sweep_v2.py test <tau>      # ONE evaluation of the approved τ on v2 test (not before approval)

Model: edge/models_v2/lgbm_classifier.onnx (FP32, the artifact the API would serve).
Data : the cached v2 prepared split (identical indices to edge/models_v2/split_indices_v2.npz).
ML path only (rule detectors are separate alert sources under C′ and are not swept).
Bootstrap: exact multinomial over (class, score-bin) cells, B = 2000, seed 0.
No source edits, no retraining. Outputs under revision-audit/reports/threshold_v2/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AUD = Path(__file__).resolve().parents[1]
ROOT = AUD.parent / "realtime-ids-healthcareBeta"
OUT = AUD / "reports" / "threshold_v2"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import onnxruntime as ort  # noqa: E402
from edge.training.label_mapper import CLASS_NAMES  # noqa: E402

CACHE = AUD / "work/v2_cache/prepared_5ebccf745c0b4863.npz"
c1 = json.loads((AUD / "reports/policy_decision/policy_comparison_results.json").read_text())
ENC_MS = c1["latency_microbench"]["ckks_encrypt_hmac_28field"]["p50_ms"]
CT_BYTES = c1["latency_microbench"]["ckks_ciphertext_b64_bytes"]
V1_RATE = 4269  # flows/s (V1 Table 6), only to scale workload as in the V1 memo
COST_RATIOS = [1, 2, 3, 5, 10, 20, 50, 100]
TAUS = [1e-4, 1e-3, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
        0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (None, None)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(float(c - h), 6), round(float(c + h), 6))


def load(part):
    z = np.load(CACHE)
    s = np.load(ROOT / "edge/models_v2/split_indices_v2.npz")
    assert np.array_equal(z[f"part_{part}"], s[part]), "cache split differs from the committed v2 split"
    X = z["X"][s[part]].astype(np.float32); y = z["y"][s[part]].astype(np.int64)
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(ROOT / "edge/models_v2/lgbm_classifier.onnx"), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    P = np.zeros((len(y), 6))
    for a in range(0, len(y), 50000):
        for r, d in enumerate(sess.run(None, {inp: X[a:a + 50000]})[1]):
            for k, v in d.items():
                P[a + r, int(k)] = v
    return y, P


def row(y, P, tau):
    s = 1 - P[:, 0]; al = s >= tau; atk = y != 0; ben = ~atk; n = len(y)
    tp, fp = int((al & atk).sum()), int((al & ben).sum()); fn, tn = int((~al & atk).sum()), int((~al & ben).sum())
    rec = tp / (tp + fn); prec = tp / (tp + fp) if tp + fp else 0.0
    r = {"tau": tau, "TP": tp, "FP": fp, "TN": tn, "FN": fn, "recall": rec, "recall_CI95": wilson(tp, tp + fn),
         "precision": prec, "F1_binary": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
         "benign_alert_pct": 100 * fp / (fp + tn), "benign_alert_CI95_pct": [100 * v for v in wilson(fp, fp + tn)],
         "all_flow_alert_pct": 100 * (tp + fp) / n, "encrypt_selected": tp + fp,
         "false_alert_share_pct": 100 * fp / (tp + fp) if tp + fp else None,
         "ckks_cores_at_4269fps": (tp + fp) / n * V1_RATE * ENC_MS / 1000,
         "ciphertext_MB_per_s_at_4269fps": (tp + fp) / n * V1_RATE * CT_BYTES / 1e6}
    for k in range(1, 6):
        m = y == k
        r[f"{CLASS_NAMES[k]}_missed"] = int((~al[m]).sum()); r[f"{CLASS_NAMES[k]}_recall"] = float(al[m].mean())
    return r


def bootstrap(y, P, B=2000):
    """Exact row bootstrap via multinomial resampling of (benign/attack × score-bin) cell counts."""
    s = 1 - P[:, 0]; atk = (y != 0).astype(int)
    edges = np.array([-np.inf] + TAUS + [np.inf])
    b = np.searchsorted(edges, s, side="right") - 1          # bin j holds TAUS[j-1] <= s < TAUS[j]
    nb = len(edges) - 1
    cells = np.bincount(atk * nb + b, minlength=2 * nb)
    rng = np.random.default_rng(0)
    boots = rng.multinomial(len(y), cells / cells.sum(), size=B).reshape(B, 2, nb)
    # alerts at TAUS[i] = bins >= i+1
    ben_ge = np.flip(np.cumsum(np.flip(boots[:, 0, :], 1), 1), 1)[:, 1:]   # FP at each τ
    atk_ge = np.flip(np.cumsum(np.flip(boots[:, 1, :], 1), 1), 1)[:, 1:]   # TP at each τ
    n_atk = boots[:, 1, :].sum(1, keepdims=True)
    fn = n_atk - atk_ge; fp = ben_ge
    prec = atk_ge / np.maximum(atk_ge + fp, 1); rec = atk_ge / n_atk
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    taus = np.array(TAUS)
    opt = taus[f1.argmax(1)]
    # cost-optimal τ for cost ratio r = cost(missed attack) / cost(false alarm); ties -> larger τ
    cost_opt = {}
    for r in COST_RATIOS:
        c = r * fn + fp
        best = taus[len(TAUS) - 1 - np.argmin(np.flip(c, 1), 1)]
        vals, cnt = np.unique(best, return_counts=True)
        cost_opt[str(r)] = {"bootstrap_mode": float(vals[cnt.argmax()]),
                            "bootstrap_CI95": [float(np.percentile(best, 2.5)), float(np.percentile(best, 97.5))],
                            "distribution": {str(v): int(k) for v, k in zip(vals, cnt)}}
    d05 = {}
    i05 = TAUS.index(0.5)
    for t in (0.3, 0.4, 0.6, 0.7, 0.8, 0.9):
        i = TAUS.index(t)
        d05[str(t)] = {"dFN_vs_0.5_CI95": [float(np.percentile(fn[:, i] - fn[:, i05], q)) for q in (2.5, 97.5)],
                       "dFP_vs_0.5_CI95": [float(np.percentile(fp[:, i] - fp[:, i05], q)) for q in (2.5, 97.5)]}
    return {"B": B, "cost_optimal_tau": cost_opt, "f1_optimal_tau_distribution": {str(t): int((opt == t).sum()) for t in TAUS if (opt == t).any()},
            "paired_differences_vs_0.5": d05}


mode = sys.argv[1]
if mode == "tune":
    y, P = load("tune")
    assert np.allclose(P.sum(1), 1, atol=1e-5)
    rows = [row(y, P, t) for t in TAUS]
    df = pd.DataFrame(rows); df.to_csv(OUT / "tune_threshold_table.csv", index=False)
    s = 1 - P[:, 0]; atk = y != 0
    edges = [0, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0000001]
    dist = pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:],
                         "benign": [int(((s >= a) & (s < b) & ~atk).sum()) for a, b in zip(edges[:-1], edges[1:])],
                         "attack": [int(((s >= a) & (s < b) & atk).sum()) for a, b in zip(edges[:-1], edges[1:])]})
    dist.to_csv(OUT / "tune_score_distribution.csv", index=False)
    am = P.argmax(1)
    cm = pd.crosstab(pd.Series(y).map(dict(enumerate(CLASS_NAMES))), pd.Series(am).map(dict(enumerate(CLASS_NAMES))))
    cm.to_csv(OUT / "tune_confusion_matrix_6class.csv")
    # calibration of s_attack on tune (reliability by decile of s among s in (0.01, 0.99))
    mid = (s > 0.01) & (s < 0.99)
    cal = []
    for a, b in [(0.01, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99)]:
        m = (s >= a) & (s < b)
        cal.append({"bin": f"[{a},{b})", "n": int(m.sum()), "mean_s": float(s[m].mean()) if m.any() else None,
                    "attack_fraction": float(atk[m].mean()) if m.any() else None})
    meta = {"split": "v2 tune", "n": int(len(y)), "benign": int((~atk).sum()), "attack": int(atk.sum()),
            "per_class": {CLASS_NAMES[k]: int((y == k).sum()) for k in range(6)},
            "argmax_classifier_fp": int(((am != 0) & ~atk).sum()),
            "argmax_equals_tau_0.5_decision": bool(np.array_equal(am != 0, s >= 0.5)),
            "flows_in_uncertain_region_0.01_0.99": int(mid.sum()),
            "calibration": cal, "bootstrap": bootstrap(y, P),
            "enc_ms_per_vector": ENC_MS, "ciphertext_bytes": CT_BYTES}
    meta["cost_optimal_tau_point"] = {}
    for r in COST_RATIOS:
        c = [r * x["FN"] + x["FP"] for x in rows]
        i = max(i for i, v in enumerate(c) if v == min(c))
        meta["cost_optimal_tau_point"][str(r)] = {"tau": TAUS[i], "FN": rows[i]["FN"], "FP": rows[i]["FP"]}
    (OUT / "tune_meta.json").write_text(json.dumps(meta, indent=1))
    cols = ["tau", "TP", "FP", "FN", "recall", "precision", "F1_binary", "benign_alert_pct", "all_flow_alert_pct",
            "MITM_missed", "DATA_EXFILTRATION_missed", "DOS_missed", "PORT_SCAN_missed", "BRUTE_FORCE_missed"]
    print(df[cols].to_string(index=False)); print(dist.to_string(index=False))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("per_class",)}, indent=1))
elif mode == "test":
    tau = float(sys.argv[2]); y, P = load("test")
    r = row(y, P, tau)
    (OUT / f"test_eval_tau_{tau}.json").write_text(json.dumps(r, indent=2, default=str))
    print(json.dumps(r, indent=1, default=str))
