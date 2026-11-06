"""
C12 — inventory of a downloaded dataset tree (CICIoMT2024) BEFORE any scoring (Comment 12 plan §4:
the label mapping is fixed before running, not after).

  python c12_inventory.py --root <CICIoMT2024 dir> --out <dir> [--jobs 4] [--no-sha]

For every capture file (.pcap / .pcapng / .cap): size, SHA-256, link type, packet count, time span,
L3/L4 mix, top source IPs / (src -> dst) pairs / destination ports by packets. Header fields only.
Bluetooth captures are listed (size, SHA-256) but not parsed: they are not IP traffic (out of scope).
Archives (.zip / .tar* / .gz / .7z / .rar) and CSV files are listed so you can see what still needs unpacking.
Writes inventory.json (+ inventory_files.csv).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c12_flowmeter import open_reader, parse  # noqa: E402

CAPTURE = {".pcap", ".pcapng", ".cap"}
ARCHIVE = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".bz2", ".xz"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def is_bluetooth(p: Path) -> bool:
    return "bluetooth" in str(p).lower() or "ble" in [s.lower() for s in p.parts]


def scan(args):
    p, root, do_sha = args
    t0 = time.time()
    d = {"file": str(p.relative_to(root)), "bytes": p.stat().st_size}
    try:
        if do_sha:
            d["sha256"] = sha256(p)
        if is_bluetooth(p):
            d["skipped"] = "bluetooth (not IP traffic)"
            return d
        fh, rd, lt = open_reader(p)
        d["linktype"] = lt
        n, kinds, src, pair, dport = 0, Counter(), Counter(), Counter(), Counter()
        first = last = None
        try:
            for ts, buf in rd:
                n += 1
                first = ts if first is None else first; last = ts
                r = parse(buf, lt, include_icmp=True)
                if isinstance(r, str):
                    kinds[r] += 1
                    continue
                s, dst, sp, dp, pr, *_ = r
                kinds[{6: "tcp", 17: "udp", 1: "icmp", 58: "icmpv6"}.get(pr, str(pr))] += 1
                src[s] += 1; pair[f"{s}->{dst}"] += 1; dport[f"{pr}/{dp}"] += 1
        finally:
            fh.close()
        d.update({"packets": n, "first_ts": first, "last_ts": last,
                  "duration_s": (last - first) if first is not None else 0, "kinds": dict(kinds),
                  "unique_src_ips": len(src), "top_src_ips": src.most_common(10),
                  "top_pairs": pair.most_common(10), "top_dst_ports": dport.most_common(10)})
    except Exception as e:  # noqa: BLE001
        d["error"] = f"{type(e).__name__}: {e}"
    d["seconds"] = round(time.time() - t0, 1)
    print(f"[{time.strftime('%H:%M:%S')}] {d['file']}: {d.get('packets', d.get('skipped', d.get('error')))}", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--no-sha", action="store_true")
    ap.add_argument("--include", action="append", default=[],
                    help="fnmatch glob on the path relative to --root; only matching captures are hashed and parsed "
                         "(repeatable). Every file is still LISTED with its size.")
    a = ap.parse_args()
    root = Path(a.root).resolve(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    import fnmatch
    allcaps = [p for p in files if p.suffix.lower() in CAPTURE]
    caps = [p for p in allcaps   # case-insensitive (the Kaggle copy uses "pcap", the CIC site "PCAP")
            if not a.include or any(fnmatch.fnmatch(str(p.relative_to(root)).lower(), g.lower()) for g in a.include)]
    not_scanned = [{"file": str(p.relative_to(root)), "bytes": p.stat().st_size} for p in allcaps if p not in caps]
    other = [{"file": str(p.relative_to(root)), "bytes": p.stat().st_size,
              "kind": "archive" if p.suffix.lower() in ARCHIVE else "csv" if p.suffix.lower() == ".csv" else "other"}
             for p in files if p.suffix.lower() not in CAPTURE]
    print(f"{len(caps)} capture files ({sum(p.stat().st_size for p in caps) / 1e9:.2f} GB), {len(other)} other files", flush=True)
    with Pool(a.jobs) as pool:
        rows = pool.map(scan, [(p, root, not a.no_sha) for p in caps], chunksize=1)
    res = {"root": str(root), "include": a.include, "captures": rows, "other_files": other,
           "captures_not_scanned": not_scanned,
           "totals": {"capture_files": len(caps), "capture_bytes": sum(r["bytes"] for r in rows),
                      "packets": sum(r.get("packets", 0) for r in rows),
                      "bluetooth_skipped": sum("skipped" in r for r in rows), "errors": sum("error" in r for r in rows),
                      "not_scanned_files": len(not_scanned), "not_scanned_bytes": sum(r["bytes"] for r in not_scanned)}}
    (out / "inventory.json").write_text(json.dumps(res, indent=1))
    import pandas as pd
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("top_") and k != "kinds"} for r in rows]).to_csv(
        out / "inventory_files.csv", index=False)
    print(json.dumps(res["totals"]), flush=True)


if __name__ == "__main__":
    main()
