"""
Diagnostic B (Comment 13 X2 anomaly): why do 2+ uvicorn workers add a constant ≈ 40 ms per request?

  python diag_b.py --repo <repo> --x2dir <dir with x2_run.py, x2_receiver.py> --sample <npz> --out <dir>

The deployed app (full system: SHAP every flow, CKKS relay to a local receiver) is timed server-side
(diag_app: X-Server-Ms = handler + framework time; X-Worker-Pid). Client latency − server time = wait
outside the handler (network stack, accept/dispatch, event-loop queueing).
Cases (15 s each, open loop, real normal-mix flows, random source IPs):
  C1  1 worker  (uvicorn --workers 1, CPU 0)                         40, 80 req/s     baseline
  C2  2 workers (uvicorn --workers 2, CPUs 0,2)                       40, 120 req/s    reproduces B?
  C3  2 independent single-worker servers (CPU 0 :8300, CPU 2 :8301), client round-robin   40, 120 req/s
  C4  2 workers, client without keep-alive (new TCP connection per request)               40 req/s
  C5  2 workers, h11 HTTP parser instead of httptools                                     40 req/s
  C6  2 workers, strictly sequential client (1 request in flight)                         15 s
  C7  2 workers, each worker restricted to its own CPU after start (taskset -p)          40, 120 req/s
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_T0 = time.time()


def say(m):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.0f}s] {m}", flush=True)


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


async def _load(urls, bodies, rate, duration, keepalive, sequential):
    import httpx
    recs = []
    lim = httpx.Limits(max_connections=512, max_keepalive_connections=512 if keepalive else 0)
    headers = {} if keepalive else {"Connection": "close"}
    async with httpx.AsyncClient(limits=lim, timeout=15.0, headers=headers) as c:
        async def one(k, ts):
            t1 = time.perf_counter()
            try:
                r = await c.post(urls[k % len(urls)], json=bodies[k % len(bodies)])
                t2 = time.perf_counter()
                recs.append((ts, t1, t2, r.status_code, float(r.headers.get("x-server-ms", "nan")),
                             r.headers.get("x-worker-pid", "")))
            except Exception:  # noqa: BLE001
                recs.append((ts, t1, time.perf_counter(), -1, float("nan"), ""))
        t0 = time.perf_counter() + 0.3
        if sequential:
            k = 0
            while time.perf_counter() < t0 + duration:
                await one(k, time.perf_counter()); k += 1
        else:
            tasks = []
            for k in range(int(rate * duration)):
                ts = t0 + k / rate
                d = ts - time.perf_counter()
                if d > 0:
                    await asyncio.sleep(d)
                tasks.append(asyncio.create_task(one(k, ts)))
            await asyncio.gather(*tasks)
    return recs


DURATION = 15.0
RATE_SCALE = 1.0


def run_case(label, urls, bodies, rate, duration=None, keepalive=True, sequential=False):
    duration = duration or DURATION; rate = rate * RATE_SCALE
    recs = asyncio.run(_load(urls, bodies, rate, duration, keepalive, sequential))
    ok = [r for r in recs if r[3] == 200]
    cl = np.array([(r[2] - r[1]) * 1e3 for r in ok]); sv = np.array([r[4] for r in ok])
    gap = cl - sv
    pids = {}
    for r in ok:
        pids[r[5]] = pids.get(r[5], 0) + 1
    out = {"label": label, "rate": rate, "sequential": sequential, "keepalive": keepalive, "requests": len(recs),
           "ok": len(ok), "errors": len(recs) - len(ok),
           "client_ms": {"p50": pct(cl, 50), "p95": pct(cl, 95), "p99": pct(cl, 99)},
           "server_ms": {"p50": pct(sv, 50), "p95": pct(sv, 95), "p99": pct(sv, 99)},
           "outside_handler_ms": {"p50": pct(gap, 50), "p95": pct(gap, 95), "p99": pct(gap, 99)},
           "requests_per_worker": pids,
           "gap_histogram_ms": {f"{a}-{b}": int(((gap >= a) & (gap < b)).sum()) for a, b in
                                ((0, 1), (1, 5), (5, 20), (20, 35), (35, 45), (45, 60), (60, 1e9))}}
    say(f"  {label:44s} {('seq' if sequential else f'{rate:.0f}/s'):>6} ok {len(ok):5d} err {out['errors']} | client p50/p99 "
        f"{out['client_ms']['p50']:6.1f}/{out['client_ms']['p99']:6.1f} | server p50/p99 {out['server_ms']['p50']:6.1f}/"
        f"{out['server_ms']['p99']:6.1f} | outside p50/p99 {out['outside_handler_ms']['p50']:6.1f}/{out['outside_handler_ms']['p99']:6.1f} ms"
        f" | workers {sorted(pids.values())}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--x2dir", required=True)
    ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 s steps, rates / 4")
    a = ap.parse_args()
    global DURATION, RATE_SCALE
    if a.quick:
        DURATION, RATE_SCALE = 3.0, 0.25
    repo, x2 = Path(a.repo).resolve(), Path(a.x2dir).resolve()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(x2)); sys.path.insert(0, str(repo))
    import x2_run as R
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    dst = out / "diag_b.json"
    args = argparse.Namespace(repo=repo, out=out)
    from contracts.flow_feature_vector import FEATURE_NAMES
    z = np.load(a.sample); X = z["X"]
    rng = np.random.default_rng(7)
    bodies = []
    for i in range(len(X)):
        meta = {k: float(v) for k, v in zip(FEATURE_NAMES, X[i])}
        meta["Destination Port"] = int(round(float(X[i, 31]) * 65535)); meta["Protocol"] = 0
        a_, b_ = rng.integers(0, 256, 2)
        bodies.append({"raw_packet": "aabbccdd", "source_ip": f"10.{a_}.{b_}.1", "metadata": meta})
    cores = R.physical_cores()
    keys = out / "keys"
    if not (keys / "public.ctx").exists():
        subprocess.run([sys.executable, "-m", "edge.crypto.keygen", "--out", str(keys)], cwd=repo, check=True,
                       env=dict(os.environ, PYTHONPATH=str(repo)))
    env = {"ALERT_WEBHOOK_URL": f"http://127.0.0.1:{R.RPORT}/alerts", "CKKS_PUBLIC_CONTEXT": str(keys / "public.ctx"),
           "CKKS_HMAC_SECRET": R.SECRET}
    R.HERE = here                                     # PYTHONPATH for the servers includes diag_app
    rx = subprocess.Popen(["taskset", "-c", ",".join(map(str, cores[-1])), sys.executable, str(x2 / "x2_receiver.py"),
                           "--repo", str(repo), "--public-context", str(keys / "public.ctx"), "--port", str(R.RPORT)],
                          env=dict(os.environ, CKKS_HMAC_SECRET=R.SECRET), start_new_session=True)
    time.sleep(3)
    client_cpus = {c for core in cores[2:-1] for c in core}
    os.sched_setaffinity(0, client_cpus)
    gov = open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip() if \
        Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").exists() else "?"
    res = {"meta": {"cores": cores, "client_cpus": sorted(client_cpus), "governor": gov,
                    "repo_commit": os.popen(f"git -C {repo} rev-parse HEAD").read().strip()}, "cases": []}
    c0, c2 = cores[0][0], cores[1][0]
    u = lambda port: f"http://127.0.0.1:{port}/v1/scan"  # noqa: E731

    def server(cpus, workers, port, extra_args=()):
        R.PORT = port
        e = dict(os.environ, PYTHONPATH=f"{repo}:{here}", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", **env)
        cmd = ["taskset", "-c", ",".join(map(str, cpus)), sys.executable, "-m", "uvicorn", "diag_app:create", "--factory",
               "--host", "127.0.0.1", "--port", str(port), "--workers", str(workers), "--log-level", "warning", *extra_args]
        p = subprocess.Popen(cmd, cwd=repo, env=e, stdout=subprocess.DEVNULL, stderr=open(out / "server_stderr.log", "a"),
                             start_new_session=True)
        for _ in range(240):
            time.sleep(1)
            try:
                if R.get_json(f"http://127.0.0.1:{port}/v1/health")["status"] in ("ok", "degraded"):
                    time.sleep(2 + 2 * workers); return p
            except Exception:  # noqa: BLE001
                if p.poll() is not None:
                    raise RuntimeError("server exited")
        raise RuntimeError("server not healthy")

    def save():
        dst.write_text(json.dumps(res, indent=1))

    try:
        say(f"cores {cores}; client CPUs {sorted(client_cpus)}; governor {gov}")
        p = server([c0], 1, 8300)
        try:
            for r in (40, 80):
                res["cases"].append(run_case("C1 1 worker", [u(8300)], bodies, r))
        finally:
            R.stop_server(p)
        save()
        p = server([c0, c2], 2, 8300)
        try:
            for r in (40, 120):
                res["cases"].append(run_case("C2 2 workers (uvicorn --workers 2)", [u(8300)], bodies, r))
            res["cases"].append(run_case("C4 2 workers, no keep-alive", [u(8300)], bodies, 40, keepalive=False))
            res["cases"].append(run_case("C6 2 workers, sequential client", [u(8300)], bodies, 0, sequential=True))
            import psutil
            kids = psutil.Process(p.pid).children(recursive=True)
            workers = [k for k in kids if "spawn_main" in " ".join(k.cmdline())]
            for k, cpu in zip(workers, (c0, c2)):
                subprocess.run(["taskset", "-a", "-cp", str(cpu), str(k.pid)], stdout=subprocess.DEVNULL)
            res["meta"]["c7_worker_pids"] = [k.pid for k in workers]
            for r in (40, 120):
                res["cases"].append(run_case("C7 2 workers, each pinned to its own CPU", [u(8300)], bodies, r))
        finally:
            R.stop_server(p)
        save()
        p = server([c0, c2], 2, 8300, ("--http", "h11"))
        try:
            res["cases"].append(run_case("C5 2 workers, h11 parser", [u(8300)], bodies, 40))
        finally:
            R.stop_server(p)
        save()
        pa = server([c0], 1, 8300); pb = server([c2], 1, 8301)
        try:
            for r in (40, 120):
                res["cases"].append(run_case("C3 2 independent servers, round-robin", [u(8300), u(8301)], bodies, r))
        finally:
            R.stop_server(pa); R.stop_server(pb)
        save()
    finally:
        os.killpg(rx.pid, signal.SIGTERM)
    res["meta"]["elapsed_s"] = round(time.time() - _T0, 1); save(); say(f"written {dst}")


if __name__ == "__main__":
    main()
