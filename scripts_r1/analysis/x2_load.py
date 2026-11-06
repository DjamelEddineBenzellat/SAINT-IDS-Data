"""
X2 open-loop HTTP load generator: replays real v2 test flows to POST /v1/scan at a fixed rate.
Each request uses a random source IP from a 65,536-address pool (the router's 100 alerts/min/IP
rate limit would otherwise drop alerts). Runs in P processes (rate/P each), pinned by the caller.
Profiles are phase lists [(seconds, "normal"|"busy"), ...]: normal = stratified test mix,
busy = attack flows only. Returns per-request records with wall-clock times.
"""
from __future__ import annotations

import asyncio
import math
import multiprocessing as mp
import time

import numpy as np


def _body(x, ip, names):
    meta = {k: float(v) for k, v in zip(names, x)}
    meta["Destination Port"] = int(round(float(x[31]) * 65535)); meta["Protocol"] = 0
    return {"raw_packet": "aabbccdd", "source_ip": ip, "metadata": meta}


def _worker(url, sample, repo, phases, rate, seed, q):
    import sys
    sys.path.insert(0, repo)
    import httpx
    from contracts.flow_feature_vector import FEATURE_NAMES
    z = np.load(sample); X, y = z["X"], z["y"]
    rng = np.random.default_rng(seed)
    pools = {"normal": np.arange(len(X)), "busy": np.where(y != 0)[0]}
    ips = [f"10.{a}.{b}.1" for a, b in rng.integers(0, 256, size=(4096, 2))]
    total = sum(d for d, _ in phases); n = int(rate * total)
    edges = np.cumsum([d for d, _ in phases])

    def body_for(k, t_rel):
        kind = phases[int(np.searchsorted(edges, t_rel, side="right")) if t_rel < total else -1][1]
        pool = pools[kind]
        return _body(X[pool[rng.integers(len(pool))]], ips[k % len(ips)], FEATURE_NAMES)

    bodies = [body_for(k, k / rate) for k in range(n)]
    recs = []

    async def main():
        t0 = time.perf_counter() + 0.5
        lim = httpx.Limits(max_connections=512, max_keepalive_connections=512)
        async with httpx.AsyncClient(limits=lim, timeout=15.0) as c:
            async def one(k, ts):
                tsend = time.perf_counter()
                try:
                    r = await c.post(url, json=bodies[k])
                    td = time.perf_counter()
                    j = r.json() if r.status_code == 200 else {}
                    recs.append((ts, tsend, td, r.status_code, j.get("alert_id", ""), bool(j.get("encrypt")),
                                 j.get("explanation_status", "")))
                except Exception:  # noqa: BLE001
                    recs.append((ts, tsend, time.perf_counter(), -1, "", False, ""))
            tasks = []
            for k in range(n):
                ts = t0 + k / rate
                d = ts - time.perf_counter()
                if d > 0:
                    await asyncio.sleep(d)
                tasks.append(asyncio.create_task(one(k, ts)))
            await asyncio.gather(*tasks)
        return t0

    wall_minus_perf = time.time() - time.perf_counter()
    t0 = asyncio.run(main())
    q.put({"recs": recs, "wall_minus_perf": wall_minus_perf, "t0": t0})


def run_load(url, sample, repo, phases, rate, max_rate_per_proc=200.0, cpus=None):
    procs = max(1, math.ceil(rate / max_rate_per_proc))
    ctx = mp.get_context("spawn"); q = ctx.Queue(); ps = []
    for i in range(procs):
        p = ctx.Process(target=_worker, args=(url, sample, repo, phases, rate / procs, 1000 + i, q))
        p.start(); ps.append(p)
        if cpus:
            try:
                import os
                os.sched_setaffinity(p.pid, cpus)
            except Exception:  # noqa: BLE001
                pass
    total = sum(d for d, _ in phases)
    outs = []
    for _ in ps:
        try:
            outs.append(q.get(timeout=total + 90))
        except Exception as e:  # noqa: BLE001 — a crashed client must not hang the run
            raise RuntimeError(f"load client did not report back (exit codes {[p.exitcode for p in ps]})") from e
    for p in ps:
        p.join()
    recs = []
    for o in outs:
        off = o["wall_minus_perf"]
        for ts, tsend, td, code, aid, enc, es in o["recs"]:
            recs.append({"t_sched": ts + off, "t_send": tsend + off, "t_done": td + off, "code": code,
                         "alert_id": aid, "encrypt": enc, "explanation_status": es})
    return recs, procs
