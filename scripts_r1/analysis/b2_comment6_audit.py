"""
B2 — Reviewer 1 Comment 6: alert-rate and encryption-fraction reconciliation.

Uses the V1 test partition reproduced by B1 (work/v1_split_reproduction.npz)
and the saved V1 models. No retraining, no threshold changes, no source edits.
Three scoring paths are evaluated:
  A. results/ablation_study.py formula (pkl LightGBM max-prob + continuous IF sigmoid)
  B. edge/inference/engine.py InferenceEngine (Eq.4: 0.7*max p + 0.3*binary IF flag; Eq.5)
  C. deployed API: edge/inference/scoring_engine.ScoringEngine + alert_models.classify_alert_level
Vectorised re-implementations of B and C are validated against the real classes
on a random sample before being applied to the full test set.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/path/to/SAINT/realtime-ids-healthcareBeta")
AUD = Path("/path/to/SAINT/revision-audit")
OUT = AUD / "reports" / "comment6"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "edge" / "models"

import joblib  # noqa: E402
import onnxruntime as ort  # noqa: E402
from edge.training.label_mapper import CLASS_NAMES  # noqa: E402

T0 = time.time()
S: dict = {}
z = np.load(AUD / "work" / "v1_split_reproduction.npz")
i_test, y_all = z["i_test"], z["y"]
X = z["X_test"].astype(np.float32)
y = y_all[i_test]
ports = z["dst_port_test"]
N = len(y)
benign = y == 0
attack = ~benign
S["test_rows"] = int(N)
S["test_benign"] = int(benign.sum())
S["test_attack"] = int(attack.sum())
S["test_attack_prevalence_pct"] = round(100 * attack.mean(), 3)

if_model = joblib.load(MODELS / "isolation_forest.pkl")
lgbm = joblib.load(MODELS / "lgbm_classifier.pkl")
sess = ort.InferenceSession(str(MODELS / "lgbm_classifier.onnx"), providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name

dec = if_model.decision_function(X)
if_pred = if_model.predict(X)                     # -1 anomaly / +1 normal
if_sig = 1.0 / (1.0 + np.exp(dec))                # scoring_engine.py:103 / ablation_study.py:63
if_flag = (if_pred == -1).astype(float)           # engine.py:290-291
p_pkl = lgbm.predict_proba(X)


def onnx_probs(Xb):
    out = []
    for s in range(0, len(Xb), 50000):
        raw = sess.run(None, {in_name: Xb[s:s + 50000]})[1]
        if isinstance(raw, list):
            arr = np.zeros((len(raw), 6), dtype=np.float32)
            for r, d in enumerate(raw):
                for k, v in d.items():
                    arr[r, int(k)] = v
        else:
            arr = np.asarray(raw, dtype=np.float32)
        out.append(arr)
    return np.concatenate(out)


p_onnx = onnx_probs(X)
S["pkl_vs_onnx_argmax_agreement"] = float(np.mean(p_pkl.argmax(1) == p_onnx.argmax(1)))
S["pkl_vs_onnx_max_abs_prob_diff"] = float(np.max(np.abs(p_pkl - p_onnx)))
print(f"[{time.time()-T0:.0f}s] model outputs computed", flush=True)


def summarize(levels: np.ndarray, cls_pred: np.ndarray, enc_levels: tuple, name: str) -> dict:
    lv = np.asarray(levels)
    alerted = np.isin(lv, ["WARNING", "CRITICAL", "CAMPAIGN"])
    enc = np.isin(lv, enc_levels)
    fp = int(np.sum((cls_pred != 0) & benign))
    tn = int(np.sum((cls_pred == 0) & benign))
    n_alert = int(alerted.sum())
    return {
        "path": name,
        "level_counts": {k: int(np.sum(lv == k)) for k in ["INFO", "WARNING", "CRITICAL", "CAMPAIGN"]},
        "classification_fpr": {"value": round(fp / (fp + tn), 6), "num": fp, "den": fp + tn,
                               "def": "argmax!=BENIGN among true BENIGN"},
        "alert_fraction_all_flows": {"value": round(n_alert / N, 6), "num": n_alert, "den": int(N)},
        "alert_fraction_benign_flows": {"value": round(int((alerted & benign).sum()) / int(benign.sum()), 6),
                                        "num": int((alerted & benign).sum()), "den": int(benign.sum())},
        "alert_fraction_attack_flows": {"value": round(int((alerted & attack).sum()) / int(attack.sum()), 6),
                                        "num": int((alerted & attack).sum()), "den": int(attack.sum())},
        "false_positive_share_of_alerts": {"value": round(int((alerted & benign).sum()) / n_alert, 6) if n_alert else None,
                                           "num": int((alerted & benign).sum()), "den": n_alert},
        "encryption_or_relay_selected": {"levels": list(enc_levels), "value": round(int(enc.sum()) / N, 6),
                                         "num": int(enc.sum()), "den": int(N)},
    }


# ── Path A: ablation formula (reproduce Table 10) ────────────────────────────
abl = json.loads((ROOT / "results" / "ablation_study.json").read_text())
pathA = []
pred_pkl = lgbm.classes_[p_pkl.argmax(1)]
maxp_pkl = p_pkl.max(1)
for cfg in abl["configurations"]:
    s = cfg["lgbm_weight"] * maxp_pkl + cfg["if_weight"] * if_sig
    lv = np.where(s >= 0.85, "CRITICAL", np.where(s >= 0.50, "WARNING", "INFO"))
    r = summarize(lv, pred_pkl, ("WARNING", "CRITICAL"), f"A_ablation_{cfg['label']}")
    ab_fpr = int(np.sum((lv != "INFO") & benign)) / int(benign.sum())   # ablation_study.py:140-141
    r["reproduces_ablation_alert_level_fpr"] = {"recomputed": round(ab_fpr, 6), "json": cfg["alert_level_fpr"],
                                                "match": round(ab_fpr, 6) == cfg["alert_level_fpr"]}
    r["weights"] = [cfg["lgbm_weight"], cfg["if_weight"]]
    pathA.append(r)
S["path_A_ablation_formula"] = pathA

# ── Path B: InferenceEngine (Eq. 4–5), vectorised ───────────────────────────
maxp_onnx = p_onnx.max(1)
pred_onnx = p_onnx.argmax(1)
sB = np.clip(0.7 * maxp_onnx + 0.3 * if_flag, 0, 1)
lvB = np.where(sB >= 0.85, "CRITICAL", np.where(sB >= 0.50, "WARNING", "INFO"))
S["path_B_engine_eq4"] = summarize(lvB, pred_onnx, ("WARNING", "CRITICAL"), "B_engine_eq4")

# ── Path C: deployed API ScoringEngine + classify_alert_level, vectorised ───
from edge.inference.alert_models import classify_alert_level  # noqa: E402

cls = pred_onnx.copy()
conf = maxp_onnx.astype(float).copy()
an = if_sig.astype(float).copy()
aflag = (an > 0.5) & (cls != 0)
hc = np.isin(ports, [2575, 104, 11112])
r1 = hc & (X[:, 0] > 4000) & (X[:, 1] < 5)
r2 = ~r1 & hc & (X[:, 2] > 150) & (X[:, 1] <= 2) & (X[:, 0] <= 4000)
cls[r1], conf[r1], an[r1], aflag[r1] = 4, 0.99, 0.99, True
cls[r2], conf[r2], an[r2], aflag[r2] = 3, 0.95, 0.95, True
names = np.array(CLASS_NAMES)[cls]
lvC = np.array([classify_alert_level(predicted_class=n, confidence=float(c), anomaly_flag=bool(f), anomaly_score=float(a)).value
                for n, c, f, a in zip(names, conf, aflag, an)])
S["path_C_api"] = summarize(lvC, cls, ("WARNING", "CRITICAL", "CAMPAIGN"), "C_api_scoring_engine")
S["path_C_api"]["fast_filter_rule_hits"] = {"rule1_DOS": int(r1.sum()), "rule2_MITM": int(r2.sum())}
S["path_C_api"]["note"] = ("raw_packet rule overlay (handlers.py) not applicable: dataset rows have no raw packet; "
                           "evasion features absent (None) so evasion override never fires; "
                           "encryption is not implemented on this path, so the count is the cloud-relay selection "
                           "(alert_router.py: WARNING/CRITICAL/CAMPAIGN).")
print(f"[{time.time()-T0:.0f}s] paths computed", flush=True)

# ── Validation of vectorised B and C against the real classes ──────────────
rng = np.random.RandomState(7)
samp = np.concatenate([rng.choice(np.where(benign)[0], 600, replace=False),
                       rng.choice(np.where(attack)[0], 400, replace=False),
                       np.where(r1 | r2)[0][:200]])
from contracts.flow_feature_vector import FlowFeatureVector as ContractFFV  # noqa: E402
from edge.collector.feature_assembler import FlowEnvelope  # noqa: E402
from edge.inference.engine import InferenceEngine  # noqa: E402
from edge.inference.flow_ingestor import FlowFeatureVector as IngestFFV  # noqa: E402
from edge.inference.scoring_engine import ScoringEngine  # noqa: E402

eng = InferenceEngine(models_dir=MODELS, enable_hot_reload=False, enable_shap=False)
se = ScoringEngine(models_dir=MODELS)
okB = okC = 0
mismatch = []
for i in samp:
    env = FlowEnvelope(feature_vector=ContractFFV.from_numpy(X[i]), hashed_src_ip="h", hashed_dst_ip="h")
    rb = eng.infer(env)
    if rb.alert_level == lvB[i] and abs(rb.anomaly_score - sB[i]) < 1e-5:
        okB += 1
    elif len(mismatch) < 5:
        mismatch.append({"path": "B", "i": int(i), "real": [rb.alert_level, rb.anomaly_score], "vec": [str(lvB[i]), float(sB[i])]})
    v = IngestFFV(features=X[i].copy(), source_ip="0.0.0.0", dest_ip="0.0.0.0", source_port=0,
                  dest_port=int(ports[i]), protocol=0, timestamp="", raw_field_count=0)
    sc = se.score(v)
    real_lv = classify_alert_level(sc.predicted_class, sc.confidence, sc.anomaly_flag, sc.anomaly_score).value
    if real_lv == lvC[i] and sc.predicted_class == names[i] and abs(sc.anomaly_score - an[i]) < 1e-5:
        okC += 1
    elif len(mismatch) < 10:
        mismatch.append({"path": "C", "i": int(i), "real": [real_lv, sc.predicted_class, sc.anomaly_score],
                         "vec": [str(lvC[i]), str(names[i]), float(an[i])]})
eng.stop()
S["validation_sample_size"] = int(len(samp))
S["validation_path_B_agree"] = okB
S["validation_path_C_agree"] = okC
S["validation_mismatches"] = mismatch

# ── Diagnostics: why BENIGN flows alert ─────────────────────────────────────
q = [0.01, 0.05, 0.25, 0.5, 0.75]
S["benign_max_prob_quantiles_onnx"] = {str(k): round(float(np.quantile(maxp_onnx[benign], k)), 6) for k in q}
S["benign_flows_predicted_benign_pct"] = round(100 * float(np.mean(pred_onnx[benign] == 0)), 4)
S["benign_if_flag_rate_pct"] = round(100 * float(if_flag[benign].mean()), 4)
S["attack_if_flag_rate_pct"] = round(100 * float(if_flag[attack].mean()), 4)
S["benign_if_sigmoid_quantiles"] = {str(k): round(float(np.quantile(if_sig[benign], k)), 6) for k in [0.01, 0.5, 0.99]}
S["eq4_warning_condition_without_if"] = "0.7*max p >= 0.5  <=>  max p >= 0.714286"
S["benign_maxp_ge_0.714286_pct"] = round(100 * float(np.mean(maxp_onnx[benign] >= 0.5 / 0.7)), 4)
S["path_C_benign_level_by_rule"] = {
    "predicted_BENIGN_conf_lt_0.8_or_flag": int(np.sum(benign & (cls == 0) & (lvC == "WARNING"))),
    "predicted_attack": int(np.sum(benign & (cls != 0))),
}
S["elapsed_s"] = round(time.time() - T0, 1)
(OUT / "comment6_metric_reconciliation.json").write_text(json.dumps(S, indent=2, default=str))
print(f"[{time.time()-T0:.0f}s] done", flush=True)
