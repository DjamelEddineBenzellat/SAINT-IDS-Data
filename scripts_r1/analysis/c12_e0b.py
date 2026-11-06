"""
C12 E0b — why does the re-extracted Friday disagree with the published CICIDS-2017 CSV? (diagnostic)

  python c12_e0b.py --repo <repo> --pcap Friday-WorkingHours.pcap --labelled <dir> --work <dir> --out <dir>

E0 (first run) found ~70 % mean column agreement and PortScan recall 99.99 % -> 0.6 % on re-extracted
features. E0b re-extracts with directional / per-packet variants and, on the matched pairs, tests which
DEFINITION of each disputed column reproduces the published value:
  flag counts : both directions | forward only | backward only | first packet | last packet | zero (each also binary)
  byte totals : L4 payload | L4 segment (header + payload) | IP length | frame (wire) length
  Flow Bytes/s: the same byte definitions over the duration
Agreement is reported overall, per label, and on the subset where both sides have the same forward and
backward packet counts (same packet set -> isolates definition differences from flow-splitting differences).
Also: where the unmatched published flows went, 20 example disagreeing pairs per label, and a model
re-score with the best definition per column (to see how much of the recall loss it recovers).
Writes e0b.json and e0b_examples.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c12_common import Scorer, rate, say, to_x33  # noqa: E402
from c12_e0 import agreement, find_offset, load_published, match, ours_key  # noqa: E402
from c12_flowmeter import Config, extract  # noqa: E402

FLAGS = ("FIN", "SYN", "RST", "PSH", "ACK", "URG")
BITS = {"FIN": 1, "SYN": 2, "RST": 4, "PSH": 8, "ACK": 16, "URG": 32}


def flag_candidates(O: pd.DataFrame, f: str) -> dict:
    fw, bw = O[f"{f}_fwd"].to_numpy(), O[f"{f}_bwd"].to_numpy()
    c = {"both": fw + bw, "fwd": fw, "bwd": bw,
         "first_pkt": ((O.first_flags.to_numpy() & BITS[f]) > 0).astype(int),
         "last_pkt": ((O.last_flags.to_numpy() & BITS[f]) > 0).astype(int), "zero": np.zeros(len(O), int)}
    for k in ("both", "fwd", "bwd"):
        c[k + "_bin"] = (c[k] > 0).astype(int)
    return c


def byte_candidates(O: pd.DataFrame, d: str) -> dict:
    pay = O["Total Length of Fwd Packets" if d == "fwd" else "Total Length of Bwd Packets"].to_numpy()
    return {"payload": pay, "frame_payload": O[f"Total Length of {'Fwd' if d == 'fwd' else 'Bwd'} Packets_frame"].to_numpy(),
            "l4_segment": O[f"{d}_l4len"].to_numpy(), "ip_length": O[f"{d}_iplen"].to_numpy(),
            "wire": O[f"{d}_wire"].to_numpy()}


def per_label(P, cand_vals, col, labels, same):
    out = {}
    for name, v in cand_vals.items():
        a = P[col].to_numpy()
        d = {"all": agreement(a, v)["agree_pct"], "same_packet_set": agreement(a[same], v[same])["agree_pct"]}
        for lab in labels:
            m = (P.Label == lab).to_numpy()
            d[lab] = agreement(a[m], v[m])["agree_pct"] if m.any() else None
        out[name] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("repo", "pcap", "labelled", "work", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--labelled-glob", default="Friday*.csv")
    ap.add_argument("--fin-mode", default="fin_existing")
    a = ap.parse_args()
    repo = Path(a.repo).resolve(); sys.path.insert(0, str(repo))
    work, out = Path(a.work), Path(a.out); work.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    res = {"repo_commit": os.popen(f"git -C {repo} rev-parse HEAD").read().strip()}
    pub = load_published(Path(a.labelled), a.labelled_glob)
    cache = work / f"e0b_flows_{a.fin_mode}_v2.pkl"
    res["fin_mode"] = a.fin_mode
    if cache.exists():
        ours = pd.read_pickle(cache); say(f"cached {len(ours):,} flows")
    else:
        t0 = time.time()
        ours, st = extract(Path(a.pcap), Config(fin_mode=a.fin_mode),
                           progress=lambda s, n1, n2: say(f"    {s.packets:,} packets"))
        ours.to_pickle(cache); say(f"extracted {len(ours):,} flows in {time.time() - t0:.0f} s")
    ours["key"] = ours_key(ours)
    off = find_offset(pub, ours); m = match(pub, ours, off["offset_hours"])
    res["clock"] = off; res["matched"] = int(len(m)); res["published"] = int(len(pub))
    P = pub.loc[m.pub_idx.to_numpy()].reset_index(drop=True)
    O = ours.loc[m.our_idx.to_numpy()].reset_index(drop=True)
    labels = sorted(P.Label.unique())
    same = ((P["Total Fwd Packets"].to_numpy() == O["Total Fwd Packets"].to_numpy())
            & (P["Total Backward Packets"].to_numpy() == O["Total Backward Packets"].to_numpy()))
    res["same_packet_set_pct"] = {"all": 100 * float(same.mean()),
                                  **{lab: 100 * float(same[(P.Label == lab).to_numpy()].mean()) for lab in labels}}
    say(f"same packet set on {res['same_packet_set_pct']}")

    # where did the unmatched published flows go?
    un = pub.drop(index=m.pub_idx.to_numpy())
    keys = set(ours.key)
    res["unmatched"] = {"rows": int(len(un)), "key_absent_in_extracted_pct": 100 * float((~un.key.isin(keys)).mean()),
                        "by_label": un.Label.value_counts().to_dict(),
                        "protocol_counts": un.Protocol.value_counts().head(5).to_dict()}

    # definition tests
    res["flags"] = {}
    for f in FLAGS:
        col = f"{f} Flag Count"
        if col not in P:
            continue
        res["flags"][f] = per_label(P, flag_candidates(O, f), col, labels, same)
        best = max(res["flags"][f], key=lambda k: res["flags"][f][k]["same_packet_set"])
        say(f"  {col}: best on same packet set = {best} {res['flags'][f][best]}")
    res["bytes"] = {}
    for d, col in (("fwd", "Total Length of Fwd Packets"), ("bwd", "Total Length of Bwd Packets")):
        res["bytes"][d] = per_label(P, byte_candidates(O, d), col, labels, same)
        best = max(res["bytes"][d], key=lambda k: res["bytes"][d][k]["same_packet_set"])
        say(f"  {col}: best = {best} {res['bytes'][d][best]}")
    dur = O["Flow Duration"].to_numpy() / 1e6
    with np.errstate(divide="ignore", invalid="ignore"):
        bps = {k: (byte_candidates(O, "fwd")[k] + byte_candidates(O, "bwd")[k]) / dur for k in ("payload", "frame_payload", "l4_segment", "ip_length", "wire")}
    res["flow_bytes_s"] = per_label(P, bps, "Flow Bytes/s", labels, same)
    for col in ("Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Flow IAT Mean", "Flow IAT Max",
                "Flow IAT Min", "Flow IAT Std", "Init_Win_bytes_forward", "Flow Packets/s"):
        res.setdefault("other_columns", {})[col] = per_label(P, {"current": O[col].to_numpy()}, col, labels, same)["current"]

    # examples
    show = ["Label", "Source Port", "Destination Port", "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
            "PSH Flag Count", "ACK Flag Count", "Init_Win_bytes_forward", "Flow IAT Max", "Flow IAT Min", "Flow IAT Mean"]
    rows = []
    rng = np.random.default_rng(0)
    for lab in labels:
        # remaining disagreements under the E0b rules (frame payload, FIN/ACK of the first packet, SYN = RST = 0)
        ok = np.ones(len(P), bool)
        for pc, ov in (("Flow Duration", O["Flow Duration"]), ("Total Fwd Packets", O["Total Fwd Packets"]),
                       ("Total Backward Packets", O["Total Backward Packets"]),
                       ("Total Length of Fwd Packets", O["Total Length of Fwd Packets_frame"]),
                       ("Total Length of Bwd Packets", O["Total Length of Bwd Packets_frame"]),
                       ("FIN Flag Count", (O.first_flags & 1 > 0).astype(int)), ("ACK Flag Count", (O.first_flags & 16 > 0).astype(int)),
                       ("Flow IAT Max", O["Flow IAT Max"])):
            ok &= np.isclose(P[pc].to_numpy(float), ov.to_numpy(float), rtol=1e-6, atol=0, equal_nan=True)
        idx = np.where((P.Label == lab).to_numpy() & ~ok)[0]
        res.setdefault("remaining_disagreement_pct", {})[lab] = 100 * float((~ok[(P.Label == lab).to_numpy()]).mean())
        for i in rng.choice(idx, min(20, len(idx)), replace=False) if len(idx) else []:
            r = {"side": "published", **{c: P.at[i, c] for c in show if c in P}}
            q = {"side": "extracted", "Label": lab, **{c: O.at[i, c] for c in show if c in O and c != "Label"}}
            q.update({f"{f}_fwd/bwd": f"{O.at[i, f + '_fwd']}/{O.at[i, f + '_bwd']}" for f in ("FIN", "SYN", "RST", "PSH", "ACK")})
            q.update({"first_flags": O.at[i, "first_flags"], "last_flags": O.at[i, "last_flags"],
                      "fwd_iplen": O.at[i, "fwd_iplen"], "fwd_wire": O.at[i, "fwd_wire"], "pair": i,
                      "fwd_frame_payload": O.at[i, "Total Length of Fwd Packets_frame"],
                      "bwd_frame_payload": O.at[i, "Total Length of Bwd Packets_frame"]})
            r["pair"] = i
            rows += [r, q]
    pd.DataFrame(rows).to_csv(out / "e0b_examples.csv", index=False)

    # re-score with the best definition per column (chosen on the same-packet-set subset)
    O2 = O.copy()
    chosen = {}
    for f in ("FIN", "SYN", "RST", "ACK"):
        k = max(res["flags"][f], key=lambda c: res["flags"][f][c]["same_packet_set"]); chosen[f] = k
        O2[f"{f} Flag Count"] = flag_candidates(O, f)[k]
    for d, col in (("fwd", "Total Length of Fwd Packets"), ("bwd", "Total Length of Bwd Packets")):
        k = max(res["bytes"][d], key=lambda c: res["bytes"][d][c]["same_packet_set"]); chosen[col] = k
        O2[col] = byte_candidates(O, d)[k]
    kb = max(res["flow_bytes_s"], key=lambda c: res["flow_bytes_s"][c]["same_packet_set"]); chosen["Flow Bytes/s"] = kb
    O2["Flow Bytes/s"] = bps[kb]
    res["best_definitions"] = chosen
    from edge.training.label_mapper import CLASS_NAMES, map_single_label
    y = np.array([CLASS_NAMES[map_single_label(str(v))] for v in P.Label]); atk = y != "BENIGN"
    sc = Scorer()
    dport, proto = P["Destination Port"].to_numpy(), P["Protocol"].to_numpy()
    rs = {}
    for name, df in (("published", P), ("extracted_e0", O), ("extracted_best_definitions", O2)):
        r = sc.score(to_x33(df, repo), dport, proto, every=0)
        ml = r.ml_alert.to_numpy()
        rs[name] = {"recall": rate(ml[atk].sum(), atk.sum()), "fpr": rate(ml[~atk].sum(), (~atk).sum()),
                    "per_class_alert_pct": {c: 100 * float(ml[y == c].mean()) for c in np.unique(y)}}
        say(f"  {name}: {rs[name]['per_class_alert_pct']}")
    res["rescore"] = rs
    (out / "e0b.json").write_text(json.dumps(res, indent=1, default=str))
    say("written e0b.json, e0b_examples.csv")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
