"""
C12 self-test (seconds): builds a small synthetic pcap (TCP with FIN, VLAN, IPv6, UDP, ICMP, ARP), checks
the extractor's flow logic and that its 6 evasion statistics equal edge/collector/evasion_detector.py.
  python c12_selftest.py --repo <repo> --tmp <dir>
"""
from __future__ import annotations

import argparse
import random
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def make_pcap(path: Path) -> int:
    import dpkt
    random.seed(0)
    t0 = 1499432400.0
    pk = []

    def eth(ip, v6=False, vlan=False):
        b = bytes(dpkt.ethernet.Ethernet(src=b"\0" * 6, dst=b"\x11" * 6, data=ip,
                                         type=dpkt.ethernet.ETH_TYPE_IP6 if v6 else dpkt.ethernet.ETH_TYPE_IP))
        return b[:12] + b"\x81\x00\x00\x05" + b[12:] if vlan else b

    def tcp(s, d, sp, dp, fl, pay=b"", win=8192, v6=False):
        t = dpkt.tcp.TCP(sport=sp, dport=dp, flags=fl, win=win, data=pay)
        if v6:
            ip = dpkt.ip6.IP6(src=socket.inet_pton(socket.AF_INET6, s), dst=socket.inet_pton(socket.AF_INET6, d), nxt=6, hlim=64, data=t)
            ip.plen = len(bytes(t))
        else:
            ip = dpkt.ip.IP(src=socket.inet_aton(s), dst=socket.inet_aton(d), p=6, data=t); ip.len = 20 + len(bytes(t))
        return ip
    T = dpkt.tcp
    t = t0
    for i in range(40):
        c, sp, dp = "192.168.10.%d" % (5 + i % 7), 40000 + i, [104, 2575, 443, 80, 11112][i % 5]
        t += random.uniform(0.5, 3)
        pk += [(t, eth(tcp(c, "192.168.10.50", sp, dp, T.TH_SYN))), (t + .001, eth(tcp("192.168.10.50", c, dp, sp, T.TH_SYN | T.TH_ACK, win=29200)))]
        tt = t + .002
        for _ in range(random.randint(1, 8)):
            tt += random.uniform(1e-4, .05); pk.append((tt, eth(tcp(c, "192.168.10.50", sp, dp, T.TH_ACK | T.TH_PUSH, bytes(random.randint(0, 1400))), vlan=i % 3 == 0)))
            tt += random.uniform(1e-4, .05); pk.append((tt, eth(tcp("192.168.10.50", c, dp, sp, T.TH_ACK, bytes(random.randint(0, 600))))))
        tt += .01
        pk += [(tt, eth(tcp(c, "192.168.10.50", sp, dp, T.TH_FIN | T.TH_ACK))), (tt + .001, eth(tcp("192.168.10.50", c, dp, sp, T.TH_ACK)))]
    for i in range(20):
        u = dpkt.udp.UDP(sport=5353 + i, dport=53, data=bytes(40 + i)); u.ulen = 8 + len(u.data)
        ip = dpkt.ip.IP(src=socket.inet_aton("192.168.10.3"), dst=socket.inet_aton("8.8.8.8"), p=17, data=u); ip.len = 20 + len(bytes(u))
        pk.append((t0 + 5 + i * 2, eth(ip)))
    for i in range(5):
        pk.append((t0 + 7 + i, eth(tcp("fd00::1", "fd00::2", 50000 + i, 2575, T.TH_SYN, v6=True), v6=True)))
        ic = dpkt.icmp.ICMP(type=8, data=dpkt.icmp.ICMP.Echo(id=1, seq=i, data=b"x" * 56))
        ip = dpkt.ip.IP(src=socket.inet_aton("10.0.0.9"), dst=socket.inet_aton("10.0.0.1"), p=1, data=ic); ip.len = 20 + len(bytes(ic))
        pk.append((t0 + 8 + i * .1, eth(ip)))
        pk.append((t0 + 9 + i, bytes(dpkt.ethernet.Ethernet(type=dpkt.ethernet.ETH_TYPE_ARP, data=b"\0" * 28))))
    with open(path, "wb") as f:
        w = dpkt.pcap.Writer(f)
        for ts, b in sorted(pk):
            w.writepkt(b, ts=ts)
    return len(pk)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--tmp", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo).resolve()))
    from c12_flowmeter import EVASION_COLS, Config, FlowMeter, _evasion, extract, open_reader, parse
    from edge.collector.capture import FlowKey, FlowRecord, RawPacketMeta
    from edge.collector.evasion_detector import EvasionDetector
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True); p = tmp / "selftest.pcap"
    n = make_pcap(p)
    expect = {("first_fin", False): (105, {6: 85, 17: 20}), ("first_fin", True): (106, {6: 85, 17: 20, 1: 1}),
              ("none", False): (65, {6: 45, 17: 20}), ("none", True): (66, {6: 45, 17: 20, 1: 1}),
              ("fin_existing", False): (105, {6: 85, 17: 20})}
    for (fm, ic), (nf, protos) in expect.items():
        df, st = extract(p, Config(fin_mode=fm, include_icmp=ic))
        assert len(df) == nf and df.Protocol.value_counts().to_dict() == protos, (fm, ic, len(df), df.Protocol.value_counts().to_dict())
        assert st.skipped["arp"] == 5
    fh, rd, lt = open_reader(p); fm = FlowMeter(Config())
    for ts, buf in rd:
        r = parse(buf, lt, False)
        if not isinstance(r, str):
            fm.packet(int(round(ts * 1e6)), *r)
    fh.close(); fm.finish()
    det, worst = EvasionDetector(), 0.0
    for f in fm.done:
        sizes = list(f[16])
        rec = FlowRecord(key=FlowKey(hashed_src_ip="a", hashed_dst_ip="b", src_port=f[19], dst_port=f[20], protocol=f[21]),
                         packets=[RawPacketMeta(hashed_src_ip="a", hashed_dst_ip="b", src_port=1, dst_port=2, protocol=f[21],
                                                payload_size=s, timestamp_ns=i * 1000) for i, s in enumerate(sizes)])
        ref = det.extract(rec).evasion_features
        for name, v in zip(EVASION_COLS, _evasion(f[16], 0.0, f[21])):
            if name != "inter_arrival_cv":
                worst = max(worst, abs(getattr(ref, name) - v))
    assert worst < 1e-9, worst
    # frame payload counts Ethernet trailer padding (CICFlowMeter-V3); IP payload does not
    import dpkt
    t = dpkt.tcp.TCP(sport=1, dport=2, flags=dpkt.tcp.TH_ACK)
    ip = dpkt.ip.IP(src=b"\x01\x02\x03\x04", dst=b"\x05\x06\x07\x08", p=6, data=t); ip.len = 40
    frame = bytes(dpkt.ethernet.Ethernet(src=b"\0" * 6, dst=b"\0" * 6, type=0x0800, data=ip)) + b"\0" * 6
    r = parse(frame, 1, False)
    assert r[5] == 0 and r[10] == 6, r
    # fin_existing: a FIN that starts a flow does not close it; a FIN on an existing flow does
    fm2 = FlowMeter(Config(fin_mode="fin_existing"))
    fm2.packet(0, "a", "b", 1, 2, 6, 0, 0x11, 100); fm2.packet(10, "b", "a", 2, 1, 6, 0, 0x10, 100)
    fm2.packet(20, "a", "b", 1, 2, 6, 0, 0x11, 100); fm2.finish()
    assert len(fm2.done) == 1 and fm2.done[0][2] + fm2.done[0][3] == 3, fm2.done
    print(f"selftest OK: {n} packets, flow counts as expected, evasion statistics == evasion_detector (max diff {worst:.1e}), "
          "frame-payload padding and fin_existing rules OK")


if __name__ == "__main__":
    main()
