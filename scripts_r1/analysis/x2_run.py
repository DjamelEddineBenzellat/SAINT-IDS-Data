"""
X2 + X3 orchestrator (Comment 13): real HTTP load tests of the full pipeline with CKKS encryption,
and the component-cost comparison L0..L3. Run on the workstation by run_x2.sh.

  python x2_run.py --repo <repo> --sample <latency_sample_v2.npz> --out <dir> [--quick]

Core placement: server workers on distinct physical cores (one logical CPU each); load generator and
receiver on the remaining physical cores. W in {1, 2, 4} (W = 6 would share cores with the client).
X2 (deployed app edge.api.gateway:app, W workers):
  ramp   normal mix at W x {20, 40, 60, 80, 100} req/s, 20 s each (stops after 2 saturated steps)
  busy   all-attack flows, 30 s, at 70 % of the highest sustainable ramp rate
  burst  2 x (30 s normal + 10 s all-attack) at the same rate
X3 (x2_app factory, W = 1): L0, L1, L2, L3 ramps.
Sustainable = 0 errors, service p99 <= 50 ms, client lag p99 <= 20 ms, 0 encrypted alerts lost.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from x2_load import run_load  # noqa: E402

_T0 = time.time()
PORT, RPORT = 8300, 9300
SECRET = "x2-load-test-hmac-secret"


def say(m):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.0f}s] {m}", flush=True)


def physical_cores():
    cores = {}
    for c in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        try:
            key = (c.joinpath("topology/physical_package_id").read_text().strip(),
                   c.joinpath("topology/core_id").read_text().strip())
        except OSError:
            continue
        cores.setdefault(key, []).append(int(c.name[3:]))
    return list(cores.values())          # [[cpu, sibling], ...]


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def get_json(url, data=None, timeout=5):
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


class CpuSampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True); self.pid = pid; self.samples = []; self.rss = []; self._halt = threading.Event()

    def run(self):
        import psutil
        try:
            root = psutil.Process(self.pid)
        except psutil.Error:
            return
        procs = {}
        while not self._halt.is_set():
            try:
                for p in [root] + root.children(recursive=True):
                    procs.setdefault(p.pid, p)
                tot, rss = 0.0, 0
                for p in list(procs.values()):
                    try:
                        tot += p.cpu_percent(None); rss += p.memory_info().rss
                    except psutil.Error:
                        procs.pop(p.pid, None)
                self.samples.append(tot); self.rss.append(rss)
            except psutil.Error:
                pass
            time.sleep(1.0)

    def stop(self):
        self._halt.set(); self.join(3)
        s = self.samples[1:] or [0.0]
        return {"cpu_percent_mean": float(np.mean(s)), "cpu_percent_max": float(np.max(s)),
                "rss_mb_max": float(max(self.rss or [0]) / 1e6)}


def start_server(args, cpus, workers, env_extra, app="edge.api.gateway:app", factory=False):
    env = dict(os.environ, PYTHONPATH=f"{args.repo}:{HERE}", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", **env_extra)
    cmd = ["taskset", "-c", ",".join(map(str, cpus)), sys.executable, "-m", "uvicorn", app, "--host", "127.0.0.1",
           "--port", str(PORT), "--workers", str(workers), "--log-level", "warning"] + (["--factory"] if factory else [])
    p = subprocess.Popen(cmd, cwd=args.repo, env=env, stdout=subprocess.DEVNULL, stderr=open(args.out / "server_stderr.log", "a"),
                         start_new_session=True)
    for _ in range(240):
        time.sleep(1)
        try:
            h = get_json(f"http://127.0.0.1:{PORT}/v1/health")
            if h.get("status") in ("ok", "degraded"):
                time.sleep(2 + workers)        # let every worker finish loading
                return p, get_json(f"http://127.0.0.1:{PORT}/v1/health")
        except Exception:  # noqa: BLE001
            if p.poll() is not None:
                raise RuntimeError("server exited; see server_stderr.log")
    raise RuntimeError("server did not become healthy")


def stop_server(p):
    try:
        os.killpg(p.pid, signal.SIGTERM); p.wait(30)
    except Exception:  # noqa: BLE001
        os.killpg(p.pid, signal.SIGKILL)


def step(args, label, phases, rate, client_cpus, server_pid, encrypted):
    get_json(f"http://127.0.0.1:{RPORT}/reset", data=b"")
    cs = CpuSampler(server_pid); cs.start()
    t = time.time()
    recs, procs = run_load(f"http://127.0.0.1:{PORT}/v1/scan", str(args.sample), str(args.repo), phases, rate,
                           cpus=client_cpus)
    wall = time.time() - t
    cpu = cs.stop()
    time.sleep(1.5)                                        # > flush window (250 ms) + encryption + send
    rx = get_json(f"http://127.0.0.1:{RPORT}/dump")
    ok = [r for r in recs if r["code"] == 200]
    svc = np.array([(r["t_done"] - r["t_send"]) * 1e3 for r in ok])
    e2e = np.array([(r["t_done"] - r["t_sched"]) * 1e3 for r in ok])
    lag = np.array([(r["t_send"] - r["t_sched"]) * 1e3 for r in recs])
    enc_ids = [r for r in ok if r["encrypt"]]
    rec = rx["receipts"]
    dl = np.array([(rec[r["alert_id"]] - r["t_done"]) * 1e3 for r in enc_ids if r["alert_id"] in rec])
    lost = sum(1 for r in enc_ids if r["alert_id"] not in rec) if encrypted else None
    dur = sum(d for d, _ in phases)
    out = {"label": label, "offered_rate": rate, "duration_s": dur, "client_procs": procs, "requests": len(recs),
           "ok": len(ok), "errors": len(recs) - len(ok), "achieved_rate": len(ok) / dur,
           "service_ms": {"p50": pct(svc, 50), "p95": pct(svc, 95), "p99": pct(svc, 99), "max": pct(svc, 100)},
           "end_to_end_ms": {"p50": pct(e2e, 50), "p99": pct(e2e, 99)}, "client_lag_p99_ms": pct(lag, 99),
           "alerts_encrypt_selected": len(enc_ids), "explanation_status": {}, **cpu, "wall_s": round(wall, 1)}
    for r in ok:
        out["explanation_status"][r["explanation_status"]] = out["explanation_status"].get(r["explanation_status"], 0) + 1
    if encrypted:
        out.update({"alerts_received": len(dl), "alerts_lost": lost, "batches": rx["batches"],
                    "alerts_per_batch": rx["alerts"] / rx["batches"] if rx["batches"] else None,
                    "rejected_batches": rx["rejected"], "bytes_to_receiver": rx["bytes"],
                    "bytes_per_1000_flows": rx["bytes"] / max(len(ok), 1) * 1000,
                    "delivery_ms": {"p50": pct(dl, 50), "p95": pct(dl, 95), "p99": pct(dl, 99), "max": pct(dl, 100)},
                    "receiver_aggregate_ms_total": rx["aggregate_ms"]})
    out["sustainable"] = bool(out["errors"] == 0 and (out["service_ms"]["p99"] or 1e9) <= 50
                              and (out["client_lag_p99_ms"] or 0) <= 20 and (not encrypted or lost == 0))
    d = out.get("delivery_ms", {})
    say(f"  {label:28s} {rate:6.0f} req/s -> {out['achieved_rate']:6.1f} ok/s, err {out['errors']}, "
        f"p50/p99 {out['service_ms']['p50'] or 0:6.1f}/{out['service_ms']['p99'] or 0:7.1f} ms, lag99 {out['client_lag_p99_ms']:5.1f}, "
        f"cpu {cpu['cpu_percent_mean']:5.0f}%"
        + (f", enc {len(enc_ids)} recv {len(dl)} lost {lost}, deliv p99 {d.get('p99') or 0:6.0f} ms, "
           f"{out['alerts_per_batch'] or 0:5.1f}/ct" if encrypted else "")
        + ("" if out["sustainable"] else "  [NOT sustainable]"))
    return out


def ramp(args, label, steps, client_cpus, pid, encrypted):
    res, bad = [], 0
    for r in steps:
        res.append(step(args, f"{label} ramp", [(args.step_s, "normal")], r, client_cpus, pid, encrypted))
        bad = bad + 1 if not res[-1]["sustainable"] else 0
        if bad >= 2:
            break
    ok = [x["offered_rate"] for x in res if x["sustainable"]]
    return res, (max(ok) if ok else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", default="1,2,4")
    ap.add_argument("--no-x3", action="store_true", help="skip the X3 component ramps")
    args = ap.parse_args()
    args.repo = Path(args.repo).resolve(); args.sample = Path(args.sample).resolve()
    args.out = Path(args.out).resolve(); args.out.mkdir(parents=True, exist_ok=True)
    args.step_s = 5 if args.quick else 20
    dst = args.out / "x2_x3.json"
    res = json.loads(dst.read_text()) if dst.exists() else {}
    cores = physical_cores()
    say(f"{len(cores)} physical cores: {cores}")
    keys = args.out / "keys"
    if not (keys / "public.ctx").exists():
        subprocess.run([sys.executable, "-m", "edge.crypto.keygen", "--out", str(keys)], cwd=args.repo, check=True,
                       env=dict(os.environ, PYTHONPATH=str(args.repo)))
    env_enc = {"ALERT_WEBHOOK_URL": f"http://127.0.0.1:{RPORT}/alerts", "CKKS_PUBLIC_CONTEXT": str(keys / "public.ctx"),
               "CKKS_HMAC_SECRET": SECRET, "IP_PSEUDONYM_KEY": "x2-load-test-pseudonym-key"}
    workers = [int(w) for w in args.workers.split(",") if int(w) <= len(cores) - 2]
    rx_cpus = cores[-1]
    rx = subprocess.Popen(["taskset", "-c", ",".join(map(str, rx_cpus)), sys.executable, str(HERE / "x2_receiver.py"),
                           "--repo", str(args.repo), "--public-context", str(keys / "public.ctx"), "--port", str(RPORT)],
                          env=dict(os.environ, CKKS_HMAC_SECRET=SECRET), start_new_session=True)
    time.sleep(3)
    res.setdefault("meta", {"cores": cores, "receiver_cpus": rx_cpus, "step_s": args.step_s, "quick": args.quick,
                            "repo_commit": os.popen(f"git -C {args.repo} rev-parse HEAD").read().strip()})
    try:
        for W in workers:
            key = f"X2_W{W}"
            if key in res:
                say(f"{key} done - skipping"); continue
            srv_cpus = [c[0] for c in cores[:W]]
            client_cpus = {c for core in cores[W:-1] for c in core} or set(cores[-1])
            say(f"{key}: server CPUs {srv_cpus}, client CPUs {sorted(client_cpus)}")
            p, health = start_server(args, srv_cpus, W, env_enc)
            enc = health["model_versions"].get("encryption", {})
            assert enc.get("status") == "active", enc
            try:
                rr, best = ramp(args, f"W{W}", [W * s for s in (20, 40, 60, 80, 100)], client_cpus, p.pid, True)
                rate = 0.7 * (best or W * 20)
                busy = step(args, f"W{W} busy (attacks only)", [(30 if not args.quick else 8, "busy")], rate,
                            client_cpus, p.pid, True)
                ph = [(30, "normal"), (10, "busy")] * 2 if not args.quick else [(6, "normal"), (3, "busy")]
                burst = step(args, f"W{W} burst", ph, rate, client_cpus, p.pid, True)
                res[key] = {"server_cpus": srv_cpus, "health_encryption": enc, "ramp": rr,
                            "max_sustainable_rate": best, "busy": busy, "burst": burst}
            finally:
                stop_server(p)
            dst.write_text(json.dumps(res, indent=1))
        for cfg, steps in (() if args.no_x3 else (("L0", (100, 200, 400, 600)), ("L1", (100, 200, 400, 600)),
                           ("L2", (20, 40, 60, 80, 100)), ("L3", (20, 40, 60, 80, 100)))):
            key = f"X3_{cfg}"
            if key in res:
                say(f"{key} done - skipping"); continue
            srv_cpus = [cores[0][0]]
            client_cpus = {c for core in cores[1:-1] for c in core}
            say(f"{key}: W=1 on CPU {srv_cpus}")
            p, health = start_server(args, srv_cpus, 1, dict(env_enc, X2_CONFIG=cfg), app="x2_app:create", factory=True)
            try:
                rr, best = ramp(args, cfg, steps, client_cpus, p.pid, cfg == "L3")
                res[key] = {"health_model_versions": health["model_versions"], "ramp": rr, "max_sustainable_rate": best}
            finally:
                stop_server(p)
            dst.write_text(json.dumps(res, indent=1))
    finally:
        os.killpg(rx.pid, signal.SIGTERM)
    res["meta"]["elapsed_s"] = round(time.time() - _T0, 1)
    dst.write_text(json.dumps(res, indent=1)); say(f"written {dst}")


if __name__ == "__main__":
    main()
