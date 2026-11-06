"""
C12 E1 diagnostic (no PCAP pass; reads the cached E1 flow tables and the per-flow decision table).

  python c12_e1_diag.py --work <work_c12/e1> --label-map label_map.json --decisions results_b/e1_decisions.csv.gz
                        --out <dir> [--friday <work_c12/e0c/e0c_flows_fin_existing.pkl>]

(1) Labelling: for every attack capture, the flows that touch an attacker, by who sent the first packet
    (attacker -> X  /  X -> attacker), protocol, and counterpart; and how many the E1 rule kept.
(2) Naming by the ML model: among flows the ML path alerts on, the model's top attack class
    (ml_top) vs the true SAINT class, per label (the reported class `pred` is the rule's class when a
    rule fires, so it does not measure the model's naming).
(3) Evasion rules: distribution of the 6 evasion statistics on benign flows (CICIoMT2024 benign +
    profiling, and all CICIDS-2017 Friday flows if --friday is given) against the V1 thresholds,
    and how often each rule fires.
Writes e1_diag.json.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c12_flowmeter import EVASION_COLS  # noqa: E402


def rule_for(rel, lm):
    rl = rel.lower()
    if any(fnmatch.fnmatch(rl, g.lower()) for g in lm.get("exclude_globs", [])):
        return "exclude"
    for r in lm["files"]:
        if fnmatch.fnmatch(rl, r["glob"].lower()):
            return r
    return None


# V1 evasion rules (edge/inference/rule_detectors.py), reproduced for the firing statistics only
# names / thresholds exactly as EVASION_THRESHOLDS and evasion_hits() in the repo at 1b1a948
RULES = {"fragmentation_attack": ("fragmentation_score", ">", 0.5), "slowloris_timing": ("inter_arrival_cv", ">", 1.5),
         "padding_evasion": ("payload_burst_ratio", ">", 5.0), "junk_injection": ("payload_size_entropy", "<", 0.5),
         "protocol_anomaly": ("byte_per_packet_variance", ">", 2.0), "statistical_anomaly": ("payload_size_skewness", ">", 2.0)}


def evasion_profile(df: pd.DataFrame) -> dict:
    out = {"flows": int(len(df))}
    q = [0.01, 0.1, 0.5, 0.9, 0.99]
    out["quantiles"] = {c: {str(k): float(v) for k, v in df[c].quantile(q).items()} for c in EVASION_COLS}
    for name, (col, op, thr) in RULES.items():
        x = df[col].to_numpy()
        out.setdefault("fires_pct", {})[name] = 100 * float((x > thr).mean() if op == ">" else (x < thr).mean())
    n = df["Total Fwd Packets"] + df["Total Backward Packets"]
    out["flows_by_packets"] = {"1": int((n == 1).sum()), "2": int((n == 2).sum()), "3-9": int(((n >= 3) & (n < 10)).sum()),
                               ">=10": int((n >= 10).sum())}
    for k, m in (("1-2 pkts", n <= 2), (">=10 pkts", n >= 10)):
        sub = df[m]
        out.setdefault("fires_pct_by_size", {})[k] = {name: 100 * float(((sub[col] > thr) if op == ">" else (sub[col] < thr)).mean())
                                                      for name, (col, op, thr) in RULES.items()} if len(sub) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--label-map", required=True)
    ap.add_argument("--decisions", required=True); ap.add_argument("--out", required=True); ap.add_argument("--friday")
    a = ap.parse_args()
    lm = json.loads(Path(a.label_map).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"labelling": {}, "naming_ml": {}, "evasion": {}}
    benign = []
    for pkl in sorted(Path(a.work).glob("*.pkl")):
        df = pd.read_pickle(pkl)
        if not len(df):
            continue
        rel = df["capture"].iloc[0]; r = rule_for(rel, lm)
        if r in (None, "exclude"):
            continue
        if r["label"] == "BENIGN":
            benign.append(df); continue
        A = set(r.get("attackers") or [])
        s, d = df["Source IP"].isin(A), df["Destination IP"].isin(A)
        proto_ok = df["Protocol"].isin(r["protocols"]) if r.get("protocols") else pd.Series(True, index=df.index)
        port_fwd = df["Destination Port"].isin(r["dst_ports"]) if r.get("dst_ports") else pd.Series(True, index=df.index)
        port_rev = df["Source Port"].isin(r["dst_ports"]) if r.get("dst_ports") else pd.Series(True, index=df.index)
        rows = {}
        for tag, m in (("attacker_first", s), ("victim_first", d & ~s), ("no_attacker", ~s & ~d)):
            g = df[m]
            rows[tag] = {"flows": int(len(g)), "by_protocol": g.Protocol.value_counts().head(4).to_dict(),
                         "top_counterparts": (g["Destination IP"] if tag == "attacker_first" else g["Source IP"]).value_counts().head(6).to_dict(),
                         "packets_per_flow_median": float((g["Total Fwd Packets"] + g["Total Backward Packets"]).median()) if len(g) else None}
        rows["kept_by_E1_rule"] = int((s & proto_ok & port_fwd).sum())
        rows["would_keep_either_direction"] = int(((s & proto_ok & port_fwd) | (d & ~s & proto_ok & port_rev)).sum())
        res["labelling"][rel] = rows
        print(f"{rel.split('/')[-1]:40s} attacker_first={rows['attacker_first']['flows']:>7} victim_first={rows['victim_first']['flows']:>7} "
              f"no_attacker={rows['no_attacker']['flows']:>7} kept={rows['kept_by_E1_rule']:>7} either_dir={rows['would_keep_either_direction']:>7}", flush=True)

    dec = pd.read_csv(a.decisions)
    for lab, g in dec[dec.is_attack].groupby("label"):
        ml = g[g.ml_alert]
        res["naming_ml"][lab] = {"flows": int(len(g)), "ml_alerts": int(len(ml)), "true_class": g.true_class.iloc[0],
                                 "ml_top_among_ml_alerts": ml.ml_top.value_counts().to_dict(),
                                 "ml_naming_pct": 100 * float((ml.ml_top == ml.true_class).mean()) if len(ml) and isinstance(g.true_class.iloc[0], str) else None}
    b = dec[~dec.is_attack]
    res["naming_ml"]["BENIGN_ml_false_alarms_top_class"] = b[b.ml_alert].ml_top.value_counts().to_dict()

    if benign:
        res["evasion"]["ciciomt_benign"] = evasion_profile(pd.concat(benign, ignore_index=True))
    if a.friday and Path(a.friday).exists():
        res["evasion"]["cicids2017_friday_all_flows"] = evasion_profile(pd.read_pickle(a.friday))
    for k, v in res["evasion"].items():
        print(k, v["flows"], "fires%:", {n: round(x, 1) for n, x in v["fires_pct"].items()}, flush=True)
        print("   by size:", v["fires_pct_by_size"], flush=True)
    for k, v in res["naming_ml"].items():
        print(k, v, flush=True)
    (out / "e1_diag.json").write_text(json.dumps(res, indent=1, default=str))
    print("written e1_diag.json")


if __name__ == "__main__":
    main()
