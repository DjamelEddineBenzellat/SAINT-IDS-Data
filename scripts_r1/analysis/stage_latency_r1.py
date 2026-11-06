"""
Stage latency of the deployed request path for manuscript R1 (Table 8, Figure 2, Table 1).
One core, in-process, no HTTP. Same stages and flows as the earlier E4 run (c9_latency_e4.py), on the
deployed code: encrypted relay active, keyed IP pseudonyms active, deployed ONNX Runtime session.

  python stage_latency_r1.py run --repo <repo> --sample <latency_sample_v2.npz> --out <dir>
                                 --onnx deployed|default --run N [--core 2] [--quick]
  python stage_latency_r1.py summary --out <dir> [--quick]

--onnx selects the only difference between the two conditions:
  deployed  edge.inference.ort_options.session_options(): 1 intra-op thread, no spinning
  default   default ONNX Runtime session options (what the E4 run used, before commit 069c65f)

Each run is a fresh process, so an ONNX Runtime thread pool cannot outlive its condition.

Blocks per run:
  B       model-only ONNX on the 10,000 sample flows, 1 thread, BEFORE the request-path session exists
          (machine-state reference, same as E4 block B)
  normal  request path on the first 2,000 sample flows (stratified mix)
  busy    request path on all attack flows of the sample (every flow alerts)
Stages per flow: ingest, score (ONNX + rules + decide), explain (policy method), route (log + pseudonyms
+ relay queue), total.

The relay encrypts real CKKS ciphertexts on a background thread pinned to the same core, as in the one-core
deployment; its batches go to a local counter instead of HTTP, because this benchmark measures the request
path (the HTTP load tests cover transport). Per-run test keys stay in memory and are never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

_T0 = time.time()
EXPECTED_ALERTS = {"normal_mix": 360, "busy_mix_attacks_only": 1746}   # from the E4 run on the same flows
OLD_TABLE_8 = {"ingest": 0.066, "score": 0.157, "explain": 10.95, "route": 0.120, "mean": 11.30, "p99": 11.77}
X3_IN_PATH_MEANS = {"ingest": 0.050, "score": 0.216, "explain": 8.32, "route": 0.106}


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.0f}s] {msg}", flush=True)


def stats(v):
    a = np.asarray(v)
    return {"n": int(len(a)), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(a.max()), "throughput_flows_per_s_serial": float(1000.0 / a.mean())}


def read(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def machine_state(core):
    base = f"/sys/devices/system/cpu/cpu{core}"
    model = None
    for line in read("/proc/cpuinfo", "").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    uptime = read("/proc/uptime", "0").split()[0]
    return {"cpu_model": model, "platform": platform.platform(), "cpu_count": os.cpu_count(), "core": core,
            "thread_siblings": read(f"{base}/topology/thread_siblings_list"),
            "governor": read(f"{base}/cpufreq/scaling_governor"),
            "energy_perf_preference": read(f"{base}/cpufreq/energy_performance_preference"),
            "no_turbo": read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
            "loadavg": list(os.getloadavg()), "uptime_s": float(uptime)}


def threads():
    """Per-thread CPU time, allowed CPUs and last CPU, from /proc."""
    tick = os.sysconf("SC_CLK_TCK")
    rows = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            stat = Path(f"/proc/self/task/{tid}/stat").read_text()
            status = Path(f"/proc/self/task/{tid}/status").read_text()
        except OSError:
            continue
        rest = stat[stat.rindex(")") + 2:].split()
        allowed = re.search(r"Cpus_allowed_list:\s*(\S+)", status).group(1)
        rows[int(tid)] = {"comm": stat[stat.index("(") + 1:stat.rindex(")")], "cpus_allowed": allowed,
                          "last_cpu": int(rest[36]), "cpu_s": (int(rest[11]) + int(rest[12])) / tick}
    return rows


def thread_delta(before, after, wall_s):
    main = threading.get_native_id()
    out = []
    for tid, r in after.items():
        used = r["cpu_s"] - before.get(tid, {}).get("cpu_s", 0.0)
        out.append({"main_thread": tid == main, "comm": r["comm"], "cpus_allowed": r["cpus_allowed"],
                    "last_cpu": r["last_cpu"], "cpu_s": round(used, 3), "busy_fraction": round(used / wall_s, 3)})
    return sorted(out, key=lambda x: -x["cpu_s"])


def model_only(models_dir, X, quick):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(models_dir / "lgbm_classifier.onnx"), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    n = 200 if quick else len(X)
    for i in range(min(500, n)):
        sess.run(None, {name: X[i:i + 1]})
    lat = []
    for i in range(n):
        t = time.perf_counter(); sess.run(None, {name: X[i:i + 1]}); lat.append((time.perf_counter() - t) * 1e3)
    return stats(lat)


def request_path(X, label, pol, models_dir, pub, hmac_secret, core, arrays):
    from contracts.flow_feature_vector import FEATURE_NAMES
    from edge.crypto.alert_encryption import AlertEncryptor
    from edge.crypto.encrypted_relay import EncryptedRelay
    from edge.inference.alert_router import AlertRouter
    from edge.inference.explainer import SHAPExplainer
    from edge.inference.flow_ingestor import FlowIngestor
    from edge.inference.scoring_engine import ScoringEngine

    ing, se, ex = FlowIngestor(), ScoringEngine(models_dir=models_dir, policy=pol), SHAPExplainer(models_dir=models_dir, policy=pol)
    info = se.get_model_info()
    assert info["model_binding"]["match"] is True and ex.model_binding["match"] is True, "model binding failed"
    so = se._onnx_session.get_session_options()
    session = {"intra_op_num_threads": so.intra_op_num_threads, "inter_op_num_threads": so.inter_op_num_threads,
               "providers": se._onnx_session.get_providers()}

    batches = [0]

    def sink(_batch):
        batches[0] += 1

    relay = EncryptedRelay(AlertEncryptor(public_context=pub, hmac_secret=hmac_secret), sink)
    log_dir = tempfile.mkdtemp()
    router = AlertRouter(log_dir=log_dir, max_alerts_per_minute=10**9, encrypted_relay=relay)
    pseudo = router._pseudo.status()

    flows = []
    for x in X:
        f = {k: float(v) for k, v in zip(FEATURE_NAMES, x)}
        f.update({"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 40000,
                  "dst_port": int(round(float(x[31]) * 65535)), "protocol": 0})
        flows.append(f)
    for f in flows[:100]:
        v = ing.ingest(f); s = se.score(v); ex.explain(v, s.class_index)

    st = {k: [] for k in ("ingest", "score", "explain", "route", "total")}
    status = {"ok": 0, "failed": 0}
    decisions, mhz = [], []
    freq_file = f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq"
    th0 = threads(); c0 = time.process_time(); w0 = time.perf_counter()
    for i, f in enumerate(flows):
        t0 = time.perf_counter(); v = ing.ingest(f)
        t1 = time.perf_counter(); s = se.score(v)
        t2 = time.perf_counter()
        try:
            e = ex.explain(v, s.class_index); es = "ok"
        except Exception:  # noqa: BLE001 — same handling as the API handler
            e, es = None, "failed"
        t3 = time.perf_counter(); router.route(s, e, v, es)
        t4 = time.perf_counter()
        for k, a, b in (("ingest", t0, t1), ("score", t1, t2), ("explain", t2, t3), ("route", t3, t4), ("total", t0, t4)):
            st[k].append((b - a) * 1e3)
        status[es] += 1
        decisions.append((int(bool(s.alert)), int(s.class_index)))
        if (i + 1) % 100 == 0:
            khz = read(freq_file)
            if khz:
                mhz.append(int(khz) / 1000.0)
        if (i + 1) % 500 == 0:
            say(f"    {label}: {i + 1}/{len(flows)} (mean total {np.mean(st['total']):.2f} ms, "
                f"explain {np.mean(st['explain']):.2f} ms, failures {status['failed']})")
    wall = time.perf_counter() - w0; cpu = time.process_time() - c0; th1 = threads()
    router.close()
    rs = relay.stats()
    shutil.rmtree(log_dir, ignore_errors=True)

    arrays.update({f"{label}_{k}": np.asarray(v, dtype=np.float32) for k, v in st.items()})
    out = {k: stats(v) for k, v in st.items()}
    out.update({"flows": len(flows), "alerts": sum(a for a, _ in decisions), "explanation_status": status,
                "decisions_sha256": hashlib.sha256(np.asarray(decisions, dtype=np.int32).tobytes()).hexdigest(),
                "wall_s": wall, "process_cpu_s": cpu, "cores_used": cpu / wall,
                "core_mhz": {"mean": float(np.mean(mhz)), "min": float(np.min(mhz)), "max": float(np.max(mhz)), "n": len(mhz)} if mhz else None,
                "threads": thread_delta(th0, th1, wall), "session": session, "pseudonymizer": pseudo,
                "relay": {k: v for k, v in rs.items() if not isinstance(v, (list, tuple))}, "relay_batches_sent": batches[0],
                "isolation_forest": info["isolation_forest"], "explanation_method": ex.method})
    return out


def cmd_run(a):
    os.sched_setaffinity(0, {a.core})        # first, as in E4: every later thread inherits this mask
    root = Path(a.repo).resolve(); sys.path.insert(0, str(root))
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"stage_r1_{a.onnx}_run{a.run}.json"
    if dst.exists():
        say(f"{dst.name} exists - skipping"); return
    os.environ["IP_PSEUDONYM_KEY"] = secrets.token_hex(32)
    hmac_secret = secrets.token_bytes(32)

    import lightgbm
    import onnxruntime
    import tenseal
    from edge.crypto.keygen import generate
    from edge.inference.decision import get_default_policy, serving_models_dir
    import edge.inference.ort_options as ort_options

    pol = get_default_policy(); models_dir = serving_models_dir(pol)
    described = pol.describe(); described.pop("source_path", None)
    z = np.load(a.sample); X, y = z["X"].astype(np.float32), z["y"]
    res = {"script": "stage_latency_r1.py", "onnx_mode": a.onnx, "run": a.run, "quick": a.quick,
           "repo_commit": os.popen(f"git -C {root} rev-parse HEAD").read().strip(), "policy": described,
           "sample_sha256": hashlib.sha256(Path(a.sample).read_bytes()).hexdigest(),
           "libraries": {"onnxruntime": onnxruntime.__version__, "lightgbm": lightgbm.__version__,
                         "numpy": np.__version__, "tenseal": tenseal.__version__},
           "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ORT_INTRA_OP_THREADS")},
           "machine_before": machine_state(a.core)}

    say(f"ONNX {a.onnx}, run {a.run}, core {a.core}, commit {res['repo_commit'][:7]}, policy {described.get('version')}")
    say("B model-only (1 thread; request-path session not created yet)")
    res["B_model_only_v2_fp32"] = model_only(models_dir, X, a.quick)
    say(f"  mean {res['B_model_only_v2_fp32']['mean_ms']:.4f} ms")

    if a.onnx == "default":
        ort_options.session_options = onnxruntime.SessionOptions     # the ONLY difference between the conditions

    _, pub = generate()
    arrays = {}
    n_normal, n_busy = (150, 150) if a.quick else (2000, None)
    atk = X[y != 0]
    say(f"normal mix ({n_normal} flows)")
    res["normal_mix"] = request_path(X[:n_normal], "normal", pol, models_dir, pub, hmac_secret, a.core, arrays)
    say(f"busy mix ({n_busy or len(atk)} attack flows)")
    res["busy_mix_attacks_only"] = request_path(atk[:n_busy], "busy", pol, models_dir, pub, hmac_secret, a.core, arrays)
    res["machine_after"] = machine_state(a.core)
    np.savez_compressed(out_dir / f"stage_r1_{a.onnx}_run{a.run}_per_flow_ms.npz", **arrays)
    dst.write_text(json.dumps(res, indent=1))
    n = res["normal_mix"]
    say(f"normal: explain {n['explain']['mean_ms']:.2f} ms, total {n['total']['mean_ms']:.2f} / p99 {n['total']['p99_ms']:.2f} ms, "
        f"cores used {n['cores_used']:.2f}")
    say(f"written {dst}")


def cmd_summary(a):
    out_dir = Path(a.out)
    runs = {"deployed": [], "default": []}
    for f in sorted(out_dir.glob("stage_r1_*_run*.json")):
        d = json.loads(f.read_text()); runs[d["onnx_mode"]].append(d)
    if not runs["deployed"] or not runs["default"]:
        sys.exit("results for both conditions are required")

    def med(mode, mix, stage, stat="mean_ms"):
        return statistics.median(r[mix][stage][stat] for r in runs[mode])

    def med_field(mode, mix, key):
        return statistics.median(r[mix][key] for r in runs[mode])

    lines = []
    w = lines.append
    summary = {"conditions": {}, "checks": []}
    for mix, title in (("normal_mix", "Normal mix"), ("busy_mix_attacks_only", "Attack burst")):
        w(f"\n{title} ({runs['deployed'][0][mix]['flows']} flows; median of {len(runs['deployed'])} runs, ms per flow)")
        w(f"  {'':24}{'deployed':>10}{'default':>10}{'old Table 8':>13}{'X3 means':>10}")
        for stage in ("ingest", "score", "explain", "route"):
            old = OLD_TABLE_8[stage] if mix == "normal_mix" else None
            x3 = X3_IN_PATH_MEANS[stage] if mix == "normal_mix" else None
            w(f"  {stage + ' mean':24}{med('deployed', mix, stage):>10.3f}{med('default', mix, stage):>10.3f}"
              f"{(f'{old:.3f}' if old else '-'):>13}{(f'{x3:.3f}' if x3 else '-'):>10}")
        for stat, name in (("mean_ms", "total mean"), ("p50_ms", "total p50"), ("p95_ms", "total p95"),
                           ("p99_ms", "total p99"), ("max_ms", "total max")):
            old = {"mean_ms": OLD_TABLE_8["mean"], "p99_ms": OLD_TABLE_8["p99"]}.get(stat) if mix == "normal_mix" else None
            w(f"  {name:24}{med('deployed', mix, 'total', stat):>10.2f}{med('default', mix, 'total', stat):>10.2f}"
              f"{(f'{old:.2f}' if old else '-'):>13}{'-':>10}")
        w(f"  {'serial flows/s':24}{1000 / med('deployed', mix, 'total'):>10.1f}{1000 / med('default', mix, 'total'):>10.1f}")
        w(f"  {'cores used by process':24}{med_field('deployed', mix, 'cores_used'):>10.2f}{med_field('default', mix, 'cores_used'):>10.2f}")
        mhz = {m: [r[mix]["core_mhz"]["mean"] for r in runs[m] if r[mix]["core_mhz"]] for m in runs}
        if mhz["deployed"] and mhz["default"]:
            w(f"  {'pinned core mean MHz':24}{statistics.median(mhz['deployed']):>10.0f}{statistics.median(mhz['default']):>10.0f}")
        w("  explain mean per run:  deployed " + " ".join(f"{r[mix]['explain']['mean_ms']:.2f}" for r in runs["deployed"])
          + "   default " + " ".join(f"{r[mix]['explain']['mean_ms']:.2f}" for r in runs["default"]))
        for m in runs:
            summary["conditions"].setdefault(m, {})[mix] = {
                "stages_mean_ms": {s: med(m, mix, s) for s in ("ingest", "score", "explain", "route", "total")},
                "total_ms": {s: med(m, mix, "total", s) for s in ("p50_ms", "p95_ms", "p99_ms", "max_ms")},
                "cores_used": med_field(m, mix, "cores_used")}

    w("\nModel-only reference, ms (block B, before the request-path session exists):")
    w("  deployed " + " ".join(f"{r['B_model_only_v2_fp32']['mean_ms']:.4f}" for r in runs["deployed"])
      + "   default " + " ".join(f"{r['B_model_only_v2_fp32']['mean_ms']:.4f}" for r in runs["default"]) + "   (E4: 0.0867-0.0895)")

    for m in ("default", "deployed"):
        r = runs[m][0]["normal_mix"]
        w(f"\nThreads, ONNX {m}, run {runs[m][0]['run']}, normal mix ({r['wall_s']:.1f} s): session intra_op={r['session']['intra_op_num_threads']}")
        for t in r["threads"][:9]:
            tag = "main" if t["main_thread"] else t["comm"]
            w(f"  {tag:18} allowed CPUs {t['cpus_allowed']:8} last CPU {t['last_cpu']:>2}  busy {t['busy_fraction'] * 100:5.1f}%")

    every = runs["deployed"] + runs["default"]
    m0 = every[0]["machine_before"]
    w(f"\nMachine: {m0['cpu_model']}; {m0['platform']}; core {m0['core']} (siblings {m0['thread_siblings']}); "
      f"governor {m0['governor']}; no_turbo {m0['no_turbo']}; libraries {every[0]['libraries']}")
    w(f"Code: commit {every[0]['repo_commit'][:7]}, policy {every[0]['policy'].get('version')}, sample {every[0]['sample_sha256'][:12]}")

    def check(ok, text):
        summary["checks"].append({"ok": bool(ok), "check": text})
        w(f"  {'PASS' if ok else 'FAIL'}  {text}")

    w("\nChecks:")
    check(len({r["repo_commit"] for r in every}) == 1, "one code commit across all runs")
    check(all(r["policy"].get("version") == "2.4.0" for r in every), "policy 2.4.0 in every run")
    check(len({r["sample_sha256"] for r in every}) == 1, "one sample file across all runs")
    for mix in ("normal_mix", "busy_mix_attacks_only"):
        rows = [r[mix] for r in every]
        check(all(x["explanation_status"]["failed"] == 0 and x["explanation_status"]["ok"] == x["flows"] for x in rows), f"{mix}: every flow explained")
        check(len({x["decisions_sha256"] for x in rows}) == 1, f"{mix}: identical decisions in every run and condition")
        if not a.quick:
            check(all(x["alerts"] == EXPECTED_ALERTS[mix] for x in rows), f"{mix}: alerts = {EXPECTED_ALERTS[mix]}, as in the earlier run on these flows")
        check(all(x["relay"].get("encrypted_alerts") == x["alerts"] for x in rows), f"{mix}: every alert encrypted by the relay")
        check(all(x["relay"].get("dropped_queue_full", 0) == 0 and x["relay"].get("encrypt_failures", 0) == 0 and x["relay"].get("send_failures", 0) == 0 for x in rows), f"{mix}: relay lost nothing")
        check(all(x["pseudonymizer"].get("ip_pseudonymization") == "active" for x in rows), f"{mix}: IP pseudonyms active")
        check(all(x["isolation_forest"] == "off" for x in rows), f"{mix}: Isolation Forest off")
    check(all(r["normal_mix"]["session"]["intra_op_num_threads"] == 1 for r in runs["deployed"]), "deployed: session has 1 intra-op thread")
    check(all(r["normal_mix"]["session"]["intra_op_num_threads"] == 0 for r in runs["default"]), "default: session uses ONNX Runtime defaults")

    text = "\n".join(lines) + "\n"
    print(text)
    (out_dir / "summary.txt").write_text(text)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", required=True); r.add_argument("--sample", required=True); r.add_argument("--out", required=True)
    r.add_argument("--onnx", choices=("deployed", "default"), required=True); r.add_argument("--run", type=int, required=True)
    r.add_argument("--core", type=int, default=2); r.add_argument("--quick", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True); s.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
