"""
C12 — metadata-only flow extractor for Comment 12 (E0 / E1 / E3).

Reads a pcap / pcapng file and builds bidirectional flows with CICFlowMeter semantics for the
columns SAINT's feature_mapper uses, plus the 6 SAINT evasion statistics. Only packet HEADERS are
parsed (addresses, ports, flags, window, lengths, timestamps); payload bytes are never read.

Semantics (CICFlowMeter-style; the open choices are options, chosen by E0 against the published
CICIDS-2017 CSVs):
  - flow key = 5-tuple; the first packet's source is the forward direction;
  - a packet more than `timeout_s` (120 s) after the flow START closes it and starts a new flow;
  - fin_mode "first_fin": a TCP FIN packet is added and closes the flow;
    fin_mode "fin_existing": as first_fin, but a FIN that STARTS a new flow does not close it (E0b: this is
    what CICFlowMeter-V3 did for CICIDS-2017); fin_mode "none": FIN does not close flows;
  - lengths are L4 payload bytes (IP total length - IP header - TCP/UDP header);
  - times in microseconds; Flow IAT over consecutive packets of the flow (both directions);
    "Flow IAT Std" sample (n-1), "Flow IAT Std_pop" population;
  - flag counts per packet, both directions ("<F> Flag Count"), plus binary variants ("_bin");
  - Init_Win_bytes_forward = TCP window of the first forward packet, -1 if none;
  - Flow Bytes/s and Flow Packets/s divide by duration in seconds (inf or NaN when duration is 0,
    as in the published CSVs);
  - TCP and UDP always; ICMP / ICMPv6 only with include_icmp (ports 0);
  - non-first IPv4 fragments are skipped and counted.
Evasion statistics follow edge/collector/evasion_detector.py exactly (payload sizes of all packets
of the flow, IAT CV with population std), via its own functions.
"""
from __future__ import annotations

import math
import socket
import struct
import sys
from array import array
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_U16 = struct.Struct("!H")
FIN, SYN, RST, ACK = 0x01, 0x02, 0x04, 0x10


@dataclass
class Config:
    timeout_s: float = 120.0
    fin_mode: str = "first_fin"          # "first_fin" | "fin_existing" | "none"
    include_icmp: bool = False
    sweep_every: int = 500_000           # packets between sweeps of expired flows (output-neutral)
    max_flows: int | None = None         # uniform reservoir sample of finished flows (memory bound); None = keep all
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Stats:
    packets: int = 0
    flows_total: int = 0                 # finished flows before sampling
    peak_active: int = 0
    used: int = 0
    skipped: Counter = field(default_factory=Counter)
    first_us: int | None = None
    last_us: int | None = None


def open_reader(path: Path):
    import dpkt
    f = open(path, "rb")
    magic = f.read(4); f.seek(0)
    if magic == b"\x0a\x0d\x0d\x0a":
        r = dpkt.pcapng.Reader(f)
    else:
        r = dpkt.pcap.Reader(f)
    return f, r, int(r.datalink())


def _l3(buf: bytes, linktype: int):
    """Return (ethertype-like tag, offset of the L3 header) or (None, 0)."""
    if linktype == 1:                                   # Ethernet (+ VLAN)
        if len(buf) < 14:
            return None, 0
        et, off = _U16.unpack_from(buf, 12)[0], 14
        while et in (0x8100, 0x88A8) and len(buf) >= off + 4:
            et, off = _U16.unpack_from(buf, off + 2)[0], off + 4
        return et, off
    if linktype == 113:                                 # Linux cooked SLL
        return (_U16.unpack_from(buf, 14)[0], 16) if len(buf) >= 16 else (None, 0)
    if linktype == 276:                                 # SLL2
        return (_U16.unpack_from(buf, 0)[0], 20) if len(buf) >= 20 else (None, 0)
    if linktype in (12, 14, 101, 228, 229):             # raw IP
        if not buf:
            return None, 0
        v = buf[0] >> 4
        return (0x0800 if v == 4 else 0x86DD if v == 6 else None), 0
    if linktype == 0:                                   # BSD loopback
        return (0x0800 if buf[4] >> 4 == 4 else 0x86DD), 4
    return None, 0


def parse(buf: bytes, linktype: int, include_icmp: bool):
    """Header-only parse. Returns (src, dst, sport, dport, proto, payload_len, flags, window, ip_len, l4_len,
    frame_payload) or a skip reason str. ip_len = IP header + L4 segment; l4_len = L4 header + payload;
    frame_payload = captured frame bytes after the L4 header (includes Ethernet trailer padding, as
    CICFlowMeter-V3 counts it). Payload bytes themselves are never read, only lengths."""
    et, o = _l3(buf, linktype)
    if et == 0x0800:
        if len(buf) < o + 20:
            return "short"
        ihl = (buf[o] & 0x0F) * 4
        tot = _U16.unpack_from(buf, o + 2)[0]
        frag = _U16.unpack_from(buf, o + 6)[0]
        if frag & 0x1FFF:
            return "ipv4_fragment"
        proto = buf[o + 9]
        src, dst = socket.inet_ntop(socket.AF_INET, buf[o + 12:o + 16]), socket.inet_ntop(socket.AF_INET, buf[o + 16:o + 20])
        l4, l4len = o + ihl, tot - ihl
    elif et == 0x86DD:
        if len(buf) < o + 40:
            return "short"
        proto = buf[o + 6]
        l4len = _U16.unpack_from(buf, o + 4)[0]
        src, dst = socket.inet_ntop(socket.AF_INET6, buf[o + 8:o + 24]), socket.inet_ntop(socket.AF_INET6, buf[o + 24:o + 40])
        l4 = o + 40
    elif et == 0x0806:
        return "arp"
    else:
        return "non_ip"
    if proto == 6:
        if len(buf) < l4 + 16:
            return "short"
        sport, dport = struct.unpack_from("!HH", buf, l4)
        doff = (buf[l4 + 12] >> 4) * 4
        flags = buf[l4 + 13]
        win = _U16.unpack_from(buf, l4 + 14)[0]
        return (src, dst, sport, dport, 6, max(l4len - doff, 0), flags, win, l4len + (l4 - o), l4len,
                max(len(buf) - l4 - doff, 0))
    if proto == 17:
        if len(buf) < l4 + 8:
            return "short"
        sport, dport = struct.unpack_from("!HH", buf, l4)
        return src, dst, sport, dport, 17, max(l4len - 8, 0), 0, -1, l4len + (l4 - o), l4len, max(len(buf) - l4 - 8, 0)
    if proto in (1, 58) and include_icmp:
        return src, dst, 0, 0, proto, max(l4len - 8, 0), 0, -1, l4len + (l4 - o), l4len, max(len(buf) - l4 - 8, 0)
    return f"proto_{proto}"


# flow state indices
(START, LAST, FP, BP, FB, BB, NFIN, NSYN, NRST, NACK, WIN, IN, IMEAN, IM2, IMIN, IMAX, SIZES,
 SRC, DST, SPORT, DPORT, PROTO, FFL, BFL, FIRSTF, LASTF, BYV) = range(27)
FLAG_BITS = (("FIN", 0x01), ("SYN", 0x02), ("RST", 0x04), ("PSH", 0x08), ("ACK", 0x10), ("URG", 0x20))
# BYV: [fwd_iplen, bwd_iplen, fwd_l4len, bwd_l4len, fwd_wire, bwd_wire, fwd_frame_payload, bwd_frame_payload]


class FlowMeter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.timeout_us = int(cfg.timeout_s * 1e6)
        self.active: dict = {}
        self.done: list = []
        self.n_done = 0
        import random
        self._rng = random.Random(cfg.seed)

    def _emit(self, f):
        """Keep a finished flow; beyond cfg.max_flows, Algorithm R keeps a uniform sample of all flows."""
        self.n_done += 1
        cap = self.cfg.max_flows
        if cap is None or len(self.done) < cap:
            self.done.append(f)
        else:
            j = self._rng.randrange(self.n_done)
            if j < cap:
                self.done[j] = f

    def _new(self, ts, src, dst, sport, dport, proto):
        return [ts, ts, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0.0, 0.0, math.inf, 0.0, array("l"),
                src, dst, sport, dport, proto, [0] * 6, [0] * 6, None, 0, [0] * 8]

    @staticmethod
    def _add(f, ts, fwd, plen, flags, win, iplen=0, l4len=0, fpl=0, wire=0):
        if f[FP] + f[BP] > 0:                           # IAT (Welford)
            iat = float(ts - f[LAST])
            f[IN] += 1
            d = iat - f[IMEAN]; f[IMEAN] += d / f[IN]; f[IM2] += d * (iat - f[IMEAN])
            if iat < f[IMIN]:
                f[IMIN] = iat
            if iat > f[IMAX]:
                f[IMAX] = iat
        f[LAST] = ts
        if f[FIRSTF] is None:
            f[FIRSTF] = flags
        f[LASTF] = flags
        y = f[BYV]; d = 0 if fwd else 1
        y[d] += iplen; y[2 + d] += l4len; y[4 + d] += wire; y[6 + d] += fpl
        if flags:
            fl = f[FFL] if fwd else f[BFL]
            for j, (_, bit) in enumerate(FLAG_BITS):
                if flags & bit:
                    fl[j] += 1
        if fwd:
            f[FP] += 1; f[FB] += plen
            if f[WIN] == -1 and f[PROTO] == 6:
                f[WIN] = win
        else:
            f[BP] += 1; f[BB] += plen
        if flags:
            if flags & FIN: f[NFIN] += 1  # noqa: E701
            if flags & SYN: f[NSYN] += 1  # noqa: E701
            if flags & RST: f[NRST] += 1  # noqa: E701
            if flags & ACK: f[NACK] += 1  # noqa: E701
        f[SIZES].append(plen)

    def packet(self, ts, src, dst, sport, dport, proto, plen, flags, win, iplen=0, l4len=0, fpl=0, wire=0):
        k = (src, dst, sport, dport, proto)
        f = self.active.get(k); fwd = True
        if f is None:
            kr = (dst, src, dport, sport, proto)
            f = self.active.get(kr)
            if f is not None:
                k, fwd = kr, False
        if f is not None and ts - f[START] > self.timeout_us:
            self._emit(self.active.pop(k)); f = None
        if f is None:
            k, fwd = (src, dst, sport, dport, proto), True
            f = self._new(ts, src, dst, sport, dport, proto); self.active[k] = f
        new = f[FP] + f[BP] == 0
        self._add(f, ts, fwd, plen, flags, win, iplen, l4len, fpl, wire)
        if proto == 6 and flags & FIN and (self.cfg.fin_mode == "first_fin" or (self.cfg.fin_mode == "fin_existing" and not new)):
            self._emit(self.active.pop(k))

    def sweep(self, now):
        old = [k for k, f in self.active.items() if now - f[START] > self.timeout_us]
        for k in old:
            self._emit(self.active.pop(k))

    def finish(self):
        for f in list(self.active.values()):
            self._emit(f)
        self.active.clear()


def _evasion(sizes: array, iat_cv: float, proto: int) -> tuple:
    """The 6 SAINT evasion statistics, same definitions as edge/collector/evasion_detector.py."""
    n = len(sizes)
    if n == 0:
        return (0.0,) * 6
    s = np.frombuffer(sizes, dtype=np.int64).astype(np.float64) if sizes.itemsize == 8 else np.asarray(sizes, dtype=np.float64)
    mean = float(s.mean()); mx, mn = float(s.max()), float(s.min())
    burst = mx / mean if mean else 0.0
    if n < 2:
        return 0.0, 0.0, burst, 0.0, 0.0, 0.0
    if mx == mn:
        ent = 0.0
    else:
        b = np.minimum(9, ((s - mn) / (mx - mn) * 9).astype(np.int64))
        p = np.bincount(b, minlength=10)[np.bincount(b, minlength=10) > 0] / n
        ent = float(-(p * np.log2(p)).sum())
    var = float(((s - mean) ** 2).mean())
    if n < 3:
        skew = 0.0
    else:
        std = math.sqrt(var) if var > 0 else 1.0
        skew = max(-10.0, min(10.0, float(((s - mean) ** 3).mean()) / std ** 3))
    tot = float(s.sum())
    frag = min(10.0, n * var / tot) if tot else 0.0
    base = {6: 2500.0, 17: 500.0, 1: 100.0}.get(proto, 1000.0)
    bpv = min(10.0, var / base) if mean else 0.0
    return ent, skew, burst, iat_cv, frag, bpv


EVASION_COLS = ("payload_size_entropy", "payload_size_skewness", "payload_burst_ratio",
                "inter_arrival_cv", "fragmentation_score", "byte_per_packet_variance")


def to_frame(flows: list, source: str = "") -> pd.DataFrame:
    rows = []
    for f in flows:
        dur = f[LAST] - f[START]; pk = f[FP] + f[BP]; by = f[FB] + f[BB]
        n = f[IN]
        if n:
            imean, imin, imax = f[IMEAN], f[IMIN], f[IMAX]
            istd = math.sqrt(f[IM2] / (n - 1)) if n > 1 else 0.0
            ipop = math.sqrt(f[IM2] / n)
            cv = ipop / imean if imean else 0.0
        else:
            imean = imin = imax = istd = ipop = cv = 0.0
        sec = dur / 1e6
        bps = by / sec if sec else (math.nan if by == 0 else math.inf)
        pps = pk / sec if sec else math.inf
        rows.append((f[SRC], f[SPORT], f[DST], f[DPORT], f[PROTO], f[START], dur, f[FP], f[BP], f[FB], f[BB],
                     bps, pps, imean, istd, ipop, imax, imin, f[NFIN], f[NSYN], f[NRST], f[NACK], f[WIN],
                     *_evasion(f[SIZES], cv, f[PROTO])))
    cols = ["Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "start_us",
            "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
            "Total Length of Bwd Packets", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
            "Flow IAT Std_pop", "Flow IAT Max", "Flow IAT Min", "FIN Flag Count", "SYN Flag Count",
            "RST Flag Count", "ACK Flag Count", "Init_Win_bytes_forward", *EVASION_COLS]
    df = pd.DataFrame(rows, columns=cols)
    ext = {}
    for j, (nm, _) in enumerate(FLAG_BITS):
        ext[f"{nm}_fwd"] = [f[FFL][j] for f in flows]; ext[f"{nm}_bwd"] = [f[BFL][j] for f in flows]
    ext["first_flags"] = [f[FIRSTF] or 0 for f in flows]; ext["last_flags"] = [f[LASTF] for f in flows]
    for j, nm in enumerate(("fwd_iplen", "bwd_iplen", "fwd_l4len", "bwd_l4len", "fwd_wire", "bwd_wire",
                            "Total Length of Fwd Packets_frame", "Total Length of Bwd Packets_frame")):
        ext[nm] = [f[BYV][j] for f in flows]
    df = pd.concat([df, pd.DataFrame(ext)], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fb = (df["Total Length of Fwd Packets_frame"] + df["Total Length of Bwd Packets_frame"]).to_numpy(float)
        sec = df["Flow Duration"].to_numpy(float) / 1e6
        df["Flow Bytes/s_frame"] = np.where(sec > 0, fb / np.where(sec > 0, sec, 1), np.where(fb == 0, np.nan, np.inf))
    for c in ("FIN", "SYN", "RST", "ACK"):
        df[f"{c} Flag Count_bin"] = (df[f"{c} Flag Count"] > 0).astype(np.int64)
    df["source"] = source
    return df


def extract(path: Path, cfg: Config, max_packets: int | None = None,
            t_from_us: int | None = None, t_to_us: int | None = None, progress=None) -> tuple[pd.DataFrame, Stats]:
    fh, rd, lt = open_reader(path)
    fm, st = FlowMeter(cfg), Stats()
    try:
        for ts, buf in rd:
            t = int(round(ts * 1e6))
            st.packets += 1
            if t_from_us is not None and t < t_from_us:
                continue
            if t_to_us is not None and t >= t_to_us:
                break
            r = parse(buf, lt, cfg.include_icmp)
            if isinstance(r, str):
                st.skipped[r] += 1
            else:
                st.used += 1
                if st.first_us is None:
                    st.first_us = t
                st.last_us = t
                fm.packet(t, *r, len(buf))
            if st.packets % cfg.sweep_every == 0:
                st.peak_active = max(st.peak_active, len(fm.active))
                fm.sweep(t)
                if progress:
                    progress(st, len(fm.active), len(fm.done))
            if max_packets and st.packets >= max_packets:
                break
    finally:
        fh.close()
    st.peak_active = max(st.peak_active, len(fm.active))
    fm.finish()
    st.flows_total = fm.n_done
    return to_frame(fm.done, path.name), st


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap"); ap.add_argument("--out", required=True)
    ap.add_argument("--fin-mode", default="first_fin"); ap.add_argument("--include-icmp", action="store_true")
    ap.add_argument("--max-packets", type=int)
    a = ap.parse_args()
    df, st = extract(Path(a.pcap), Config(fin_mode=a.fin_mode, include_icmp=a.include_icmp), a.max_packets)
    df.to_pickle(a.out)
    print(json.dumps({"flows": len(df), "packets": st.packets, "used": st.used, "skipped": dict(st.skipped)}),
          file=sys.stderr)
