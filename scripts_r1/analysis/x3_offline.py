"""
X3 offline part (Comment 13): detection contribution of each component under the same alert rule
(policy C′, tau = 0.25), and the in-process cost of each component per flow (one core).

  python x3_offline.py --repo <repo> --cache <dir with prepared_*.npz> --sample <latency_sample> --out <dir>

Detection: L0 = ML source only (rule detectors disabled) vs L1 = ML + rule detectors, on
  (a) the whole v2 test split (364,800 flows) and (b) the 16 demo/iot_evasion_suite.ps1 scenarios.
  SHAP (L2) and CKKS (L3) do not change any decision (tested; regression), so their detection = L1.
Cost per flow (2,000 normal-mix flows): ingest, ONNX + decide, rule detectors, SHAP, local routing,
  relay enqueue, and CKKS encryption amortized per alert (packed 117 per ciphertext).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def counts(alert, y):
    atk = y != 0
    tp, fp = int((alert & atk).sum()), int((alert & ~atk).sum()); fn, tn = int((~alert & atk).sum()), int((~alert & ~atk).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "recall": tp / (tp + fn), "benign_alert_pct": 100 * fp / (fp + tn)}


def load_split(repo, cache):
    committed = np.load(repo / "edge/models_v2/split_indices_v2.npz")
    for c in sorted(Path(cache).glob("prepared_*.npz")):
        z = np.load(c)
        if all(np.array_equal(z[f"part_{k}"], committed[k]) for k in ("train", "val", "tune", "test")):
            t = committed["test"]
            return z["X"][t].astype(np.float32), z["y"][t]
    raise SystemExit("no cache matches the committed v2 split")


def detection_test_split(repo, cache):
    import onnxruntime as ort
    from edge.inference.decision import decide, get_default_policy
    from edge.inference.rule_detectors import all_rule_hits
    X, y = load_split(repo, cache)
    sess = ort.InferenceSession(str(repo / "edge/models_v2/lgbm_classifier.onnx"), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    P = np.zeros((len(X), 6))
    for a in range(0, len(X), 50000):
        for r, d in enumerate(sess.run(None, {name: X[a:a + 50000]})[1]):
            for k, v in d.items():
                P[a + r, int(k)] = v
    pol1 = get_default_policy(); pol0 = replace(pol1, rule_detectors_enabled=False)
    a0 = np.zeros(len(X), bool); a1 = np.zeros(len(X), bool); n_rule = 0; rule_only = 0
    for i in range(len(X)):
        hits = all_rule_hits(X[i].astype(np.float64), int(round(float(X[i, 31]) * 65535)), None)
        p = P[i].tolist()
        a0[i] = decide(p, policy=pol0).alert
        d1 = decide(p, rule_hits=hits, policy=pol1); a1[i] = d1.alert
        n_rule += bool(hits); rule_only += bool(hits) and not a0[i]
    return {"n": int(len(y)), "L0_ml_only": counts(a0, y), "L1_ml_plus_rules": counts(a1, y),
            "flows_with_rule_hits": int(n_rule), "alerts_added_by_rules": int(rule_only),
            "added_by_rules_true_attacks": int((a1 & ~a0 & (y != 0)).sum()),
            "added_by_rules_benign": int((a1 & ~a0 & (y == 0)).sum())}


def detection_healthcare(repo):
    from edge.api.handlers import Handlers
    from edge.api.schemas import PacketScan
    from edge.inference.alert_router import AlertRouter
    from edge.inference.decision import get_default_policy, serving_models_dir
    from edge.inference.flow_ingestor import FlowIngestor
    from edge.inference.scoring_engine import ScoringEngine
    src = (repo / "demo/iot_evasion_suite.ps1").read_text(encoding="utf-8", errors="replace")
    out = {}
    pol1 = get_default_policy()
    for tag, pol in (("L0_ml_only", replace(pol1, rule_detectors_enabled=False)), ("L1_ml_plus_rules", pol1)):
        h = Handlers(flow_ingestor=FlowIngestor(), scoring_engine=ScoringEngine(models_dir=serving_models_dir(pol), policy=pol),
                     alert_router=AlertRouter(log_dir=tempfile.mkdtemp(), max_alerts_per_minute=10**6))
        atk = ben = det = fa = 0
        for blk in re.split(r"\n(?=Send-HealthcareTest )", src)[1:]:
            blk = blk.split("\n# ===")[0]
            g = lambda k: re.search(rf'-{k}\s+"([^"]*)"', blk).group(1)  # noqa: E731
            port = int(re.search(r"-Port\s+(\d+)", blk).group(1))
            meta = {"Destination Port": port, "Dst Port": port, "Protocol": int(re.search(r"-Protocol\s+(\d+)", blk).group(1)),
                    "Src Port": 40000}
            meta.update({m.group(1): float(m.group(2)) for m in re.finditer(r'"([A-Za-z_]+)"\s*=\s*(-?[0-9.]+)', blk)})
            r = h._scan_single(PacketScan(raw_packet=g("RawHex"), source_ip=g("IP"), metadata=meta))
            if "$true" in blk.split("-ShouldDetect")[1][:8]:
                atk += 1; det += r.alert
            else:
                ben += 1; fa += r.alert
        out[tag] = {"attacks": atk, "detected": det, "baselines": ben, "false_alarms": fa}
    return out


def cost_breakdown(repo, sample, n=2000):
    from contracts.flow_feature_vector import FEATURE_NAMES
    from edge.crypto.alert_encryption import AlertEncryptor, alert_vector
    from edge.crypto.encrypted_relay import EncryptedRelay
    from edge.crypto.keygen import generate
    from edge.inference.alert_router import AlertRouter
    from edge.inference.decision import get_default_policy, serving_models_dir
    from edge.inference.explainer import SHAPExplainer
    from edge.inference.flow_ingestor import FlowIngestor
    from edge.inference.rule_detectors import all_rule_hits
    from edge.inference.scoring_engine import ScoringEngine
    pol = get_default_policy(); d = serving_models_dir(pol)
    z = np.load(sample); X = z["X"][:n].astype(np.float32)
    ing, se, ex = FlowIngestor(), ScoringEngine(models_dir=d, policy=pol), SHAPExplainer(models_dir=d, policy=pol)
    sec, pub = generate()
    sent = []
    relay = EncryptedRelay(AlertEncryptor(public_context=pub, hmac_secret=b"x3"), sent.append, flush_ms=250)
    router = AlertRouter(log_dir=tempfile.mkdtemp(), max_alerts_per_minute=10**9, encrypted_relay=relay)
    t = {k: [] for k in ("ingest", "score_incl_rules", "rules", "shap", "route_incl_enqueue", "alert_vector")}
    n_alert = 0
    for x in X:
        f = {k: float(v) for k, v in zip(FEATURE_NAMES, x)}
        f.update({"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "dst_port": int(round(float(x[31]) * 65535)), "protocol": 0})
        a = time.perf_counter(); v = ing.ingest(f)
        b = time.perf_counter(); s = se.score(v)
        c = time.perf_counter(); all_rule_hits(v.features, v.dest_port, getattr(v, "evasion_features", None))
        dd = time.perf_counter(); e = ex.explain(v, s.class_index)
        ee = time.perf_counter(); router.route(s, e, v, "ok")
        ff = time.perf_counter(); alert_vector(s, v); gg = time.perf_counter()
        for k, (u, w) in zip(t, ((a, b), (b, c), (c, dd), (dd, ee), (ee, ff), (ff, gg))):
            t[k].append((w - u) * 1e3)
        n_alert += bool(s.alert)
    router.close()
    st = relay.stats()
    m = {k: float(np.mean(v)) for k, v in t.items()}
    enc_per_alert = st["encrypt_ms"]["mean"] / st["alerts_per_ciphertext_mean"]
    alert_share = n_alert / len(X)
    comp = {"L0_ml_only": m["ingest"] + m["score_incl_rules"] - m["rules"] + m["route_incl_enqueue"]}
    comp["L1_plus_rules"] = comp["L0_ml_only"] + m["rules"]
    comp["L2_plus_shap"] = comp["L1_plus_rules"] + m["shap"]
    comp["L3_plus_ckks_amortized"] = comp["L2_plus_shap"] + alert_share * enc_per_alert
    return {"n_flows": len(X), "alert_share": alert_share, "stage_mean_ms": m,
            "ckks": {"encrypt_ms_per_ciphertext": st["encrypt_ms"]["mean"], "alerts_per_ciphertext": st["alerts_per_ciphertext_mean"],
                     "encrypt_ms_per_alert": enc_per_alert, "relay_counters": {k: st[k] for k in ("encrypted_alerts", "ciphertexts", "dropped_queue_full")}},
            "per_flow_mean_ms_by_config": comp,
            "note": "L3 adds the background encryption cost amortized over all flows (alerts x encrypt per alert); "
                    "routing cost includes the non-blocking enqueue. Measured with every flow in one process."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--core", type=int, default=2); ap.add_argument("--skip-split", action="store_true")
    a = ap.parse_args()
    repo = Path(a.repo).resolve(); sys.path.insert(0, str(repo))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dst = out / "x3_offline.json"
    if dst.exists():
        say("x3_offline.json exists - skipping"); return
    os.sched_setaffinity(0, {a.core})
    res = {"repo_commit": os.popen(f"git -C {repo} rev-parse HEAD").read().strip()}
    say("healthcare scenarios L0 vs L1"); res["healthcare_suite"] = detection_healthcare(repo); say(f"  {res['healthcare_suite']}")
    say("cost breakdown (2,000 flows, 1 core)"); res["cost"] = cost_breakdown(repo, a.sample)
    say(f"  {res['cost']['per_flow_mean_ms_by_config']}")
    if not a.skip_split:
        say("v2 test split L0 vs L1 (364,800 flows; ~1-2 min)"); res["test_split"] = detection_test_split(repo, a.cache)
        ts = res["test_split"]; say(f"  L0 {ts['L0_ml_only']}  L1 {ts['L1_ml_plus_rules']}  rule hits {ts['flows_with_rule_hits']}")
    dst.write_text(json.dumps(res, indent=1)); say(f"written {dst}")


if __name__ == "__main__":
    import logging
    import warnings
    logging.disable(logging.WARNING); warnings.filterwarnings("ignore")
    main()
