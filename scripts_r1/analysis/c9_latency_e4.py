"""
C9 — E4: latency of the deployed path after Comment 4 step 1 (policy 2.1.0: no Isolation Forest in
the request path, native top-5 SHAP for every flow). One core, one thread, in-process, no HTTP.

  python c9_latency_e4.py --repo <repo> --sample <latency_sample_v2.npz> --out <dir> [--core 2]

Blocks:
  B       model-only ONNX FP32 on the 10,000 real sample flows (reference, same as c7 block B)
  normal  API path on the first 2,000 sample flows (stratified mix, ≈17.5 % attacks)
  busy    API path on all attack flows of the sample (alert burst: every flow alerts / is explained)
Stages per flow: ingest, score (ONNX + rules + decide), explain (policy method), route (temp log), total.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_T0 = time.time()


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.0f}s] {msg}", flush=True)


def stats(v):
    a = np.asarray(v)
    return {"n": int(len(a)), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(a.max()), "throughput_flows_per_s_serial": float(1000.0 / a.mean())}


def model_only(root, X):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(root / "edge/models_v2/lgbm_classifier.onnx"), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for i in range(500):
        sess.run(None, {name: X[i:i + 1]})
    lat = []
    for i in range(len(X)):
        t = time.perf_counter(); sess.run(None, {name: X[i:i + 1]}); lat.append((time.perf_counter() - t) * 1e3)
    return stats(lat)


def api_path(X, label):
    from contracts.flow_feature_vector import FEATURE_NAMES
    from edge.inference.alert_router import AlertRouter
    from edge.inference.decision import get_default_policy, serving_models_dir
    from edge.inference.explainer import SHAPExplainer
    from edge.inference.flow_ingestor import FlowIngestor
    from edge.inference.scoring_engine import ScoringEngine

    pol = get_default_policy(); d = serving_models_dir(pol)
    ing, se, ex = FlowIngestor(), ScoringEngine(models_dir=d, policy=pol), SHAPExplainer(models_dir=d, policy=pol)
    info = se.get_model_info()
    assert info["model_binding"]["match"] is True and ex.model_binding["match"] is True
    router = AlertRouter(log_dir=tempfile.mkdtemp(), max_alerts_per_minute=10**9)
    flows = []
    for x in X:
        f = {k: float(v) for k, v in zip(FEATURE_NAMES, x)}
        f.update({"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 40000,
                  "dst_port": int(round(float(x[31]) * 65535)), "protocol": 0})
        flows.append(f)
    for f in flows[:100]:
        v = ing.ingest(f); s = se.score(v); ex.explain(v, s.class_index)
    st = {k: [] for k in ("ingest", "score", "explain", "route", "total")}
    status = {"ok": 0, "failed": 0, "unavailable": 0}; alerts = 0; top_k = set()
    for i, f in enumerate(flows):
        t0 = time.perf_counter(); v = ing.ingest(f)
        t1 = time.perf_counter(); s = se.score(v)
        t2 = time.perf_counter()
        try:
            e = ex.explain(v, s.class_index); es = "ok"; top_k.add(len(e.top_features_local))
        except Exception:  # noqa: BLE001 — same handling as the API handler
            e, es = None, "failed"
        t3 = time.perf_counter(); router.route(s, e, v, es)
        t4 = time.perf_counter()
        for k, a, b in (("ingest", t0, t1), ("score", t1, t2), ("explain", t2, t3), ("route", t3, t4), ("total", t0, t4)):
            st[k].append((b - a) * 1e3)
        status[es] += 1; alerts += bool(s.alert)
        if (i + 1) % 500 == 0:
            say(f"    {label}: {i + 1}/{len(flows)} (mean total {np.mean(st['total']):.2f} ms, "
                f"explain {np.mean(st['explain']):.2f} ms, failures {status['failed']})")
    out = {k: stats(v) for k, v in st.items()}
    out.update({"flows": len(flows), "alerts": alerts, "explanation_status": status, "top_k_seen": sorted(top_k),
                "isolation_forest": info["isolation_forest"], "explanation_method": ex.method,
                "anomaly_fields_none": s.anomaly_score is None and s.anomaly_flag is None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--core", type=int, default=2)
    a = ap.parse_args()
    root = Path(a.repo).resolve(); sys.path.insert(0, str(root))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dst = out / f"latency_e4_{socket.gethostname()}.json"
    if dst.exists():
        say(f"{dst.name} exists - skipping"); return
    os.sched_setaffinity(0, {a.core})
    z = np.load(a.sample); X, y = z["X"].astype(np.float32), z["y"]
    from edge.inference.decision import get_default_policy
    pol = get_default_policy()
    res = {"repo_commit": os.popen(f"git -C {root} rev-parse HEAD").read().strip(), "policy": pol.describe(),
           "machine": {"host": socket.gethostname(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
                       "core": a.core}}
    say("B model-only (10,000 real flows)"); res["B_model_only_v2_fp32"] = model_only(root, X)
    say(f"  mean {res['B_model_only_v2_fp32']['mean_ms']:.4f} ms")
    say("normal mix (2,000 flows)"); res["normal_mix"] = api_path(X[:2000], "normal")
    atk = X[y != 0]
    say(f"busy mix ({len(atk)} attack flows)"); res["busy_mix_attacks_only"] = api_path(atk, "busy")
    for k in ("normal_mix", "busy_mix_attacks_only"):
        t = res[k]["total"]
        say(f"{k}: total mean {t['mean_ms']:.3f} ms, p99 {t['p99_ms']:.3f} ms; alerts {res[k]['alerts']}/{res[k]['flows']}; "
            f"explanations {res[k]['explanation_status']}")
    dst.write_text(json.dumps(res, indent=1)); say(f"written {dst}")


if __name__ == "__main__":
    import logging
    import warnings
    logging.disable(logging.WARNING); warnings.filterwarnings("ignore")
    main()
