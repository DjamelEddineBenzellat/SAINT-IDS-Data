"""
C7 — Latency benchmarks for the v2 serving model (V1 Table 6 and the end-to-end claims).

  python c7_latency_v2.py --repo <repo> --cache <dir with prepared_*.npz> --out <dir> [--threads 1]

Benchmarks (single flow per call, warm-up excluded):
  A. V1 protocol replica : onnx_exporter.benchmark_onnx_inference method — 10,000 Gaussian
                           randn vectors, default ORT threads. Reproduces how Table 6 was made.
  B. Revised model-only  : 10,000 REAL v2 test vectors (stratified sample), ORT intra-op = --threads.
  C. API path in-process : FlowIngestor.ingest → ScoringEngine.score (IF + ONNX + rules + decide)
                           [→ SHAPExplainer.explain top_k=3] → AlertRouter.route (temp log dir).
                           Per-stage times. HTTP/serialisation not included.
Models: V1 (edge/models) and v2 (edge/models_v2) on identical inputs.
Output: <out>/latency_<host>.json and <out>/latency_sample_v2.npz. No repository files are written.
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

ROOT: Path = Path(".")          # set from --repo in main()
_T0 = time.time()


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.0f}s] {msg}", flush=True)


def pin(cores):
    """Set CPU affinity for this process (and threads created afterwards)."""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cores)


def stats(lat_ms):
    a = np.asarray(lat_ms)
    return {"n": int(len(a)), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(a.max()), "throughput_flows_per_s_serial": float(1000.0 / a.mean())}


def make_sample(cache_dir: Path, sample: Path, n=10_000):
    if sample.exists():
        z = np.load(sample)
        return z["X"], z["y"]
    from sklearn.model_selection import train_test_split
    committed = np.load(ROOT / "edge/models_v2/split_indices_v2.npz")
    for c in sorted(Path(cache_dir).glob("prepared_*.npz")):
        z = np.load(c)
        if all(np.array_equal(z[f"part_{k}"], committed[k]) for k in ("train", "val", "tune", "test")):
            break
    else:
        raise SystemExit(f"no prepared_*.npz in {cache_dir} matches the committed v2 split")
    t = committed["test"]
    _, idx = train_test_split(t, test_size=n, stratify=z["y"][t], random_state=0)
    X, y = z["X"][idx].astype(np.float32), z["y"][idx]
    np.savez(sample, X=X, y=y, test_indices=idx, cache_file=str(c))
    return X, y


def onnx_loop(path, X, threads=None, warmup=500):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for i in range(min(warmup, len(X))):
        sess.run(None, {name: X[i:i + 1]})
    lat = []
    for i in range(len(X)):
        t0 = time.perf_counter(); sess.run(None, {name: X[i:i + 1]}); lat.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 2500 == 0:
            say(f"    {Path(path).parent.name}/{Path(path).name}: {i + 1}/{len(X)}")
    return stats(lat)


def api_path(models_dir, X, Y, n, shap):
    from contracts.flow_feature_vector import FEATURE_NAMES
    from edge.inference.alert_router import AlertRouter
    from edge.inference.decision import AlertPolicy
    from edge.inference.explainer import SHAPExplainer
    from edge.inference.flow_ingestor import FlowIngestor
    from edge.inference.scoring_engine import ScoringEngine

    pol = AlertPolicy.load()                         # deployed policy (τ from file); V1 dir -> binding mismatch is expected
    ing, se = FlowIngestor(), ScoringEngine(models_dir=models_dir, policy=pol)
    ex = SHAPExplainer(models_dir=models_dir) if shap else None
    router = AlertRouter(log_dir=tempfile.mkdtemp(), max_alerts_per_minute=10**9)
    flows = []
    for x in X[:n]:
        f = {k: float(v) for k, v in zip(FEATURE_NAMES, x)}
        f.update({"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 40000,
                  "dst_port": int(round(float(x[31]) * 65535)), "protocol": 0})
        flows.append(f)
    def explain(v, s):
        # Same handling as the API handler (edge/api/handlers.py): a SHAP failure is caught and the
        # alert is routed without an explanation. Failures are counted, not hidden.
        try:
            return ex.explain(v, s.class_index, top_k=3), None
        except Exception as err:  # noqa: BLE001
            return None, type(err).__name__

    for f in flows[:10 if ex else 100]:              # warm-up (short with SHAP: it is slow)
        v = ing.ingest(f); s = se.score(v)
        if ex: explain(v, s)
    st = {"ingest": [], "score": [], "shap": [], "route": [], "total": [], "if_only": []}
    fails = {"count": 0, "by_true_class": {}, "by_predicted_class": {}, "errors": {}}
    for i, f in enumerate(flows):
        t0 = time.perf_counter(); v = ing.ingest(f)
        t1 = time.perf_counter(); s = se.score(v)
        t2 = time.perf_counter(); e, err = explain(v, s) if ex else (None, None)
        if err:
            fails["count"] += 1
            for key, val in (("by_true_class", int(Y[i])), ("by_predicted_class", s.predicted_class), ("errors", err)):
                fails[key][str(val)] = fails[key].get(str(val), 0) + 1
        t3 = time.perf_counter(); router.route(s, e, v)
        t4 = time.perf_counter()
        st["ingest"].append((t1 - t0) * 1e3); st["score"].append((t2 - t1) * 1e3)
        st["shap"].append((t3 - t2) * 1e3); st["route"].append((t4 - t3) * 1e3); st["total"].append((t4 - t0) * 1e3)
        if (i + 1) % (25 if ex else 250) == 0:
            say(f"    API path {Path(models_dir).name} shap={bool(ex)}: {i + 1}/{len(flows)} "
                f"(mean total so far {np.mean(st['total']):.2f} ms"
                + (f", mean SHAP {np.mean(st['shap']):.1f} ms, SHAP failures {fails['count']})" if ex else ")"))
        x64 = v.features.reshape(1, -1).astype(np.float64)
        t5 = time.perf_counter(); se._if_model.decision_function(x64); st["if_only"].append((time.perf_counter() - t5) * 1e3)
    res = {k: stats(v) for k, v in st.items() if not (k == "shap" and not ex)}
    if ex:
        fails["rate"] = fails["count"] / len(flows)
        res["shap_failures"] = fails
    return res


def machine():
    info = {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count()}
    try:
        info["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:  # noqa: BLE001
        info["cpu_model"] = platform.processor()
    for k, p in [("no_turbo", "/sys/devices/system/cpu/intel_pstate/no_turbo"),
                 ("governor", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")]:
        try:
            info[k] = open(p).read().strip()
        except OSError:
            pass
    import onnxruntime, sklearn, lightgbm, shap
    info["versions"] = {"onnxruntime": onnxruntime.__version__, "sklearn": sklearn.__version__,
                        "lightgbm": lightgbm.__version__, "shap": shap.__version__}
    return info


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cache", required=True, help="directory holding the v2 prepared_*.npz cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--api-flows", type=int, default=2000)
    ap.add_argument("--shap-flows", type=int, default=200, help="flows for the with-SHAP API path (SHAP is slow)")
    ap.add_argument("--core", type=int, default=2, help="core for the single-thread blocks B and C")
    a = ap.parse_args()
    ROOT = Path(a.repo).resolve(); sys.path.insert(0, str(ROOT))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sample = out / "latency_sample_v2.npz"
    X, y = make_sample(Path(a.cache), sample)
    from edge.inference.decision import get_default_policy, model_binding
    pol = get_default_policy()
    res = {"machine": machine(), "ort_intra_op_threads_revised": a.threads, "sample": str(sample),
           "policy": pol.describe(), "v2_binding": model_binding(ROOT / "edge/models_v2/lgbm_classifier.onnx", pol),
           "repo_commit": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
           "sample_class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}}
    all_cores = set(range(os.cpu_count() or 1))
    res["affinity"] = {"A": "all cores (V1 method: default ORT threads)", "B_C": f"core {a.core}"}
    gauss = np.random.RandomState(42).randn(10_000, 33).astype(np.float32)
    for tag, d in [("v1", ROOT / "edge/models"), ("v2", ROOT / "edge/models_v2")]:
        for f in ("lgbm_classifier.onnx", "lgbm_classifier_int8.onnx"):
            k = f"{tag}_{'int8' if 'int8' in f else 'fp32'}"
            say(f"A {k} (Gaussian, default threads, all cores)")
            pin(all_cores)
            res.setdefault("A_v1_protocol_gaussian_default_threads", {})[k] = onnx_loop(d / f, gauss, None, warmup=10)
            say(f"B {k} (real vectors, {a.threads} thread, core {a.core})")
            pin({a.core})
            res.setdefault("B_real_vectors_model_only", {})[k] = onnx_loop(d / f, X, a.threads)
            say(f"{k} A {round(res['A_v1_protocol_gaussian_default_threads'][k]['mean_ms'], 4)} ms  "
                f"B {round(res['B_real_vectors_model_only'][k]['mean_ms'], 4)} ms")
    for tag, d in [("v1", ROOT / "edge/models"), ("v2", ROOT / "edge/models_v2")]:
        for shap_on in (False, True):
            k = f"{tag}_{'with' if shap_on else 'no'}_shap"
            n = a.shap_flows if shap_on else a.api_flows
            say(f"C {k} ({n} flows, core {a.core})")
            pin({a.core})
            res.setdefault("C_api_path_in_process", {})[k] = api_path(d, X, y, n, shap_on)
            t = res["C_api_path_in_process"][k]["total"]
            sf = res["C_api_path_in_process"][k].get("shap_failures")
            say(f"{k} total mean {t['mean_ms']:.3f} ms, p99 {t['p99_ms']:.3f} ms"
                + (f", SHAP failures {sf['count']}/{n}" if sf else ""))
    p = out / f"latency_{res['machine']['host']}.json"
    p.write_text(json.dumps(res, indent=1))
    say(f"written {p}")


if __name__ == "__main__":
    import logging
    logging.disable(logging.WARNING)
    main()
