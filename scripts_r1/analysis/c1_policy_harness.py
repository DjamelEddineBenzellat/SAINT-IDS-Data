"""
C1 — Canonical alert / encryption policy: controlled diagnostic comparison.

No source edits, no retraining, no threshold changes in the code base.
Same deployed model (FP32 ONNX), same features, same V1 partition for every policy.
Thresholds for candidate C are chosen on VALIDATION only and frozen before TEST.
Part 2 replays the 16 healthcare scenarios of demo/iot_evasion_suite.ps1 through the
real API Handlers in-process (alert log redirected to a temp dir), with/without overlay.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/path/to/SAINT/realtime-ids-healthcareBeta")
AUD = Path("/path/to/SAINT/revision-audit")
OUT = AUD / "reports" / "policy_decision"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "edge" / "models"

import joblib  # noqa: E402
import onnxruntime as ort  # noqa: E402
import edge.training.dataset_loader as dl  # noqa: E402
from edge.training.feature_mapper import map_cicids_to_features  # noqa: E402
from edge.training.label_mapper import CLASS_NAMES, map_cicids_labels  # noqa: E402
from edge.inference.alert_models import classify_alert_level  # noqa: E402

T0 = time.time()
R: dict = {"model": "edge/models/lgbm_classifier.onnx (FP32, deployed default)",
           "if_model": "edge/models/isolation_forest.pkl"}


def log(m):
    print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


# ── Data: rebuild V1 val/test from the original functions; check against B1 ──
dl.CICIDS2017_DIR = ROOT / "data" / "cicids2017"
df = dl.load_cicids2017()
ports_all = df["Destination Port"].to_numpy().astype(np.int64)
X_all = map_cicids_to_features(df)
y_all = map_cicids_labels(df["Label"])
del df
z = np.load(AUD / "work" / "v1_split_reproduction.npz")
i_val, i_test = z["i_val"], z["i_test"]
assert np.array_equal(y_all, z["y"]) and np.array_equal(X_all[i_test], z["X_test"]), "B1 split mismatch"
R["split_check_vs_B1"] = "identical"
sess = ort.InferenceSession(str(MODELS / "lgbm_classifier.onnx"), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
if_model = joblib.load(MODELS / "isolation_forest.pkl")


def onnx_probs(X):
    out = []
    for s in range(0, len(X), 50000):
        raw = sess.run(None, {inp: X[s:s + 50000]})[1]
        a = np.zeros((len(raw), 6), dtype=np.float64)
        for r, d in enumerate(raw):
            for k, v in d.items():
                a[r, int(k)] = v
        out.append(a)
    return np.concatenate(out)


SETS = {}
for name, idx in [("val", i_val), ("test", i_test)]:
    X = X_all[idx].astype(np.float32)
    P = onnx_probs(X)
    dec = if_model.decision_function(X)
    SETS[name] = dict(X=X, y=y_all[idx], port=ports_all[idx], P=P,
                      if_sig=1.0 / (1.0 + np.exp(dec)), if_flag=(if_model.predict(X) == -1))
    log(f"{name}: {len(idx)} rows scored")
del X_all

# ── Probability sanity ─────────────────────────────────────────────────────
for name, S in SETS.items():
    sums = S["P"].sum(1)
    R[f"prob_check_{name}"] = {"n_classes_output": int(S["P"].shape[1]),
                               "classes": CLASS_NAMES[:6], "row_sum_min": float(sums.min()),
                               "row_sum_max": float(sums.max()), "p_min": float(S["P"].min()),
                               "p_max": float(S["P"].max())}


# ── Policies (all produce alert bool + severity + encrypt bool) ─────────────
def policy_A(S):  # engine.py Eq.4–5
    s = np.clip(0.7 * S["P"].max(1) + 0.3 * S["if_flag"], 0, 1)
    lv = np.where(s >= 0.85, "CRITICAL", np.where(s >= 0.50, "WARNING", "INFO"))
    return lv != "INFO", lv, np.isin(lv, ["WARNING", "CRITICAL"])


def policy_B(S):  # scoring_engine + classify_alert_level (validated 1000/1000 in B2)
    cls = S["P"].argmax(1).copy(); conf = S["P"].max(1).copy(); an = S["if_sig"].copy()
    flag = (an > 0.5) & (cls != 0)
    X, port = S["X"], S["port"]
    hc = np.isin(port, [2575, 104, 11112])
    r1 = hc & (X[:, 0] > 4000) & (X[:, 1] < 5)
    r2 = ~r1 & hc & (X[:, 2] > 150) & (X[:, 1] <= 2) & (X[:, 0] <= 4000)
    cls[r1], conf[r1], an[r1], flag[r1] = 4, .99, .99, True
    cls[r2], conf[r2], an[r2], flag[r2] = 3, .95, .95, True
    names = np.array(CLASS_NAMES)[cls]
    lv = np.array([classify_alert_level(n, float(c), bool(f), float(a)).value
                   for n, c, f, a in zip(names, conf, flag, an)])
    return lv != "INFO", lv, np.isin(lv, ["WARNING", "CRITICAL", "CAMPAIGN"])


def policy_C(S, tau):
    s = 1.0 - S["P"][:, 0]
    al = s >= tau
    return al, np.where(al, "ALERT", "INFO"), al


def policy_C_or_IF(S, tau):
    al = ((1.0 - S["P"][:, 0]) >= tau) | S["if_flag"]
    return al, np.where(al, "ALERT", "INFO"), al


def policy_argmax(S):
    al = S["P"].argmax(1) != 0
    return al, np.where(al, "ALERT", "INFO"), al


def evaluate(S, al, lv, enc):
    y = S["y"]; atk = y != 0; ben = ~atk
    tp = int((al & atk).sum()); fp = int((al & ben).sum()); tn = int((~al & ben).sum()); fn = int((~al & atk).sum())
    pred = S["P"].argmax(1)
    cfp = int(((pred != 0) & ben).sum()); ctn = int(((pred == 0) & ben).sum())
    n = len(y)
    rec = tp / (tp + fn); prec = tp / (tp + fp) if tp + fp else 0.0
    per_cls = {CLASS_NAMES[k]: round(float(al[y == k].mean()), 6) for k in range(1, 6) if (y == k).any()}
    levels = {k: int(v) for k, v in zip(*np.unique(lv, return_counts=True))}
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "attack_recall": round(rec, 6), "attack_precision": round(prec, 6),
            "binary_F1_attack_positive": round(2 * prec * rec / (prec + rec), 6) if prec + rec else 0.0,
            "classifier_FPR_argmax": {"value": round(cfp / (cfp + ctn), 6), "num": cfp, "den": cfp + ctn},
            "benign_alert_rate": {"value": round(fp / (fp + tn), 6), "num": fp, "den": fp + tn},
            "all_flow_alert_rate": {"value": round((tp + fp) / n, 6), "num": tp + fp, "den": n},
            "false_alert_share": {"value": round(fp / (tp + fp), 6) if tp + fp else None, "num": fp, "den": tp + fp},
            "encryption_selected": {"value": round(int(enc.sum()) / n, 6), "num": int(enc.sum()), "den": n},
            "per_attack_class_alert_recall": per_cls, "severity_counts": levels}


# ── Validation sweep for C, threshold selection on VAL only ─────────────────
V = SETS["val"]
sV = 1.0 - V["P"][:, 0]; atkV = V["y"] != 0
grid = np.unique(np.concatenate([np.logspace(-6, 0, 121) , 1 - np.logspace(-6, -0.3, 60), [0.5]]))
grid = grid[(grid > 0) & (grid <= 1)]
curve = []
for t in grid:
    al = sV >= t
    tp = int((al & atkV).sum()); fp = int((al & ~atkV).sum()); fn = int((~al & atkV).sum())
    rec = tp / (tp + fn); prec = tp / (tp + fp) if tp + fp else 0.0
    curve.append({"tau": float(t), "recall": rec, "precision": prec,
                  "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                  "benign_alert_rate": fp / int((~atkV).sum()), "all_flow_alert_rate": (tp + fp) / len(sV)})
pd.DataFrame(curve).to_csv(OUT / "val_threshold_curve_policyC.csv", index=False)
best = max(curve, key=lambda r: r["F1"])
resB_val = evaluate(V, *policy_B(V))
b_rate = resB_val["benign_alert_rate"]["value"]
feasible = [r for r in curve if r["benign_alert_rate"] <= b_rate]
matched = max(feasible, key=lambda r: r["recall"]) if feasible else None
FROZEN = {"tau_valF1": best["tau"], "tau_0.5": 0.5, "tau_matched_to_API_val_benign_rate": matched["tau"] if matched else None,
          "selection_data": "validation only", "API_val_benign_alert_rate_used_for_matching": b_rate}
R["frozen_thresholds"] = FROZEN
log(f"frozen thresholds {FROZEN}")

# calibration of s_attack on validation (10 equal-width bins)
bins = np.linspace(0, 1, 11); b = np.clip(np.digitize(sV, bins) - 1, 0, 9)
cal = []; ece = 0.0
for k in range(10):
    m = b == k
    if m.any():
        cal.append({"bin": f"[{bins[k]:.1f},{bins[k+1]:.1f})", "n": int(m.sum()), "mean_score": float(sV[m].mean()),
                    "observed_attack_fraction": float(atkV[m].mean())})
        ece += m.mean() * abs(sV[m].mean() - atkV[m].mean())
R["calibration_val_s_attack"] = {"bins": cal, "ECE": round(float(ece), 6)}

# ── Evaluate every policy on VAL and TEST with frozen settings ──────────────
POL = {"A_paper_Eq4_5": policy_A, "B_deployed_API": policy_B, "argmax_reference": policy_argmax,
       "C_tau_valF1": lambda S: policy_C(S, FROZEN["tau_valF1"]),
       "C_tau_0.5": lambda S: policy_C(S, 0.5),
       "C_or_IFflag_tau_valF1": lambda S: policy_C_or_IF(S, FROZEN["tau_valF1"])}
if FROZEN["tau_matched_to_API_val_benign_rate"] is not None:
    POL["C_tau_matched_API"] = lambda S: policy_C(S, FROZEN["tau_matched_to_API_val_benign_rate"])
R["results"] = {}
for split, S in SETS.items():
    R["results"][split] = {}
    for pn, f in POL.items():
        R["results"][split][pn] = evaluate(S, *f(S))
    log(f"{split} evaluated")
R["test_confusion_matrix_argmax_6class"] = pd.crosstab(SETS["test"]["y"], SETS["test"]["P"].argmax(1)).values.tolist()
R["class_counts"] = {sp: {CLASS_NAMES[k]: int((S["y"] == k).sum()) for k in range(6)} for sp, S in SETS.items()}

# ── Latency micro-benchmarks (this laptop, single flow, CPU) ────────────────
x1 = SETS["test"]["X"][:1]


def bench(fn, n=2000):
    for _ in range(50): fn()
    t = []
    for _ in range(n):
        a = time.perf_counter(); fn(); t.append((time.perf_counter() - a) * 1000)
    t = np.array(t); return {"n": n, "p50_ms": round(float(np.median(t)), 4), "p99_ms": round(float(np.quantile(t, .99)), 4)}


lat = {"onnx_single_flow": bench(lambda: sess.run(None, {inp: x1})),
       "if_decision_function_single_flow": bench(lambda: if_model.decision_function(x1), 500),
       "if_predict_single_flow": bench(lambda: if_model.predict(x1), 500)}
p1 = SETS["test"]["P"][:1]
lat["policyC_decision_only"] = bench(lambda: (1.0 - p1[0, 0]) >= 0.5)
lat["policyA_decision_only"] = bench(lambda: 0.7 * p1[0].max() + 0.3 * 1.0 >= 0.5)
lat["policyB_classify_alert_level_only"] = bench(lambda: classify_alert_level("BENIGN", 0.99, False, 0.43))
# real CKKS (TenSEAL) encryption of one 28-field cloud vector via the project's own classes
try:
    from edge.crypto.ckks_encryptor import CKKSEncryptor, ts as backend
    from edge.crypto.key_manager import KeyManager
    from edge.crypto.phi_sanitizer import sanitize_array
    km = KeyManager(); enc = CKKSEncryptor(km)
    v28 = sanitize_array(SETS["test"]["X"][0])
    lat["ckks_encrypt_hmac_28field"] = bench(lambda: enc.encrypt(v28), 200)
    lat["ckks_backend"] = backend.__name__
    lat["ckks_ciphertext_b64_bytes"] = len(enc.encrypt(v28).ciphertext_b64)
except Exception as e:  # noqa: BLE001
    lat["ckks_error"] = repr(e)
R["latency_microbench"] = lat
log("latency done")

# ── Part 2: healthcare scenarios through the real API handlers ──────────────
src = (ROOT / "demo" / "iot_evasion_suite.ps1").read_text(encoding="utf-8", errors="replace")
blocks = re.split(r"\n(?=Send-HealthcareTest )", src)[1:]
scen = []
for blk in blocks:
    blk = blk.split("\n# ===")[0]
    g = lambda k: re.search(rf'-{k}\s+"([^"]*)"', blk)  # noqa: E731
    feats = {m.group(1): float(m.group(2)) for m in re.finditer(r'"([A-Za-z_]+)"\s*=\s*(-?[0-9.]+)', blk)}
    scen.append({"name": g("Name").group(1), "category": g("Category").group(1), "raw": g("RawHex").group(1),
                 "ip": g("IP").group(1), "port": int(re.search(r"-Port\s+(\d+)", blk).group(1)),
                 "proto": int(re.search(r"-Protocol\s+(\d+)", blk).group(1)), "features": feats,
                 "expected": g("ExpectedClass").group(1),
                 "should_detect": "$true" in re.search(r"-ShouldDetect\s+(\$\w+)", blk).group(1)})
assert len(scen) == 16, len(scen)

from edge.api.handlers import Handlers  # noqa: E402
from edge.api.schemas import PacketScan  # noqa: E402
from edge.inference.alert_router import AlertRouter  # noqa: E402
from edge.inference.flow_ingestor import FlowIngestor  # noqa: E402
from edge.inference.scoring_engine import ScoringEngine  # noqa: E402

tmp = tempfile.mkdtemp(prefix="saint_c1_")
ing, se = FlowIngestor(), ScoringEngine(models_dir=MODELS)


class NoOverlay(Handlers):
    def _apply_rule_overlay(self, scoring, vector, raw_len):
        return scoring


class NoOverlayNoEnrich(NoOverlay):
    @staticmethod
    def _enrich_synthetic_flow(flow, raw_len):
        return flow


variants = {"API_full": Handlers, "API_no_overlay": NoOverlay, "API_no_overlay_no_enrichment": NoOverlayNoEnrich}
hc_rows = []
for vname, cls_ in variants.items():
    h = cls_(flow_ingestor=ing, scoring_engine=se, explainer=None,
             alert_router=AlertRouter(log_dir=tmp, log_filename=f"{vname}.jsonl", max_alerts_per_minute=10**6))
    for sc in scen:
        meta = {"Destination Port": sc["port"], "Dst Port": sc["port"], "Protocol": sc["proto"], "Src Port": 40000}
        meta.update(sc["features"])
        pk = PacketScan(raw_packet=sc["raw"], source_ip=sc["ip"], metadata=meta)
        resp = h._scan_single(pk)
        vec = ing.ingest(h._decode_packet(pk))
        p = onnx_probs(vec.to_model_input())[0]
        hc_rows.append({"variant": vname, "scenario": sc["name"], "category": sc["category"],
                        "expected": sc["expected"], "should_detect": sc["should_detect"],
                        "severity": resp.severity, "predicted": resp.predicted_class,
                        "detected_API": resp.severity != "INFO",
                        "model_argmax": CLASS_NAMES[int(p.argmax())], "model_s_attack": round(float(1 - p[0]), 6),
                        "C_detected_tau_valF1": bool(1 - p[0] >= FROZEN["tau_valF1"]),
                        "C_detected_tau_0.5": bool(1 - p[0] >= 0.5)})
hc = pd.DataFrame(hc_rows)
hc.to_csv(OUT / "healthcare_scenarios_replay.csv", index=False)
summ = {}
for vname in variants:
    d = hc[hc.variant == vname]; atk = d[d.should_detect]; ben = d[~d.should_detect]
    summ[vname] = {"API_attacks_detected": f"{int(atk.detected_API.sum())}/{len(atk)}",
                   "API_attacks_class_correct": f"{int((atk.predicted == atk.expected).sum())}/{len(atk)}",
                   "API_baseline_false_alarms": f"{int(ben.detected_API.sum())}/{len(ben)}",
                   "model_only_C_tau_valF1_detected": f"{int(atk.C_detected_tau_valF1.sum())}/{len(atk)}",
                   "model_only_C_tau_valF1_false_alarms": f"{int(ben.C_detected_tau_valF1.sum())}/{len(ben)}",
                   "model_argmax_class_correct": f"{int((atk.model_argmax == atk.expected).sum())}/{len(atk)}"}
R["healthcare_replay_summary"] = summ
R["healthcare_replay_note"] = ("In-process replay of the 16 scenarios with the current code; SHAP disabled "
                               "(does not affect decisions); alert logs written to a temp dir, not the repo.")
R["elapsed_s"] = round(time.time() - T0, 1)
(OUT / "policy_comparison_results.json").write_text(json.dumps(R, indent=2, default=str))
log("done")
