"""
C12 E0 — extractor fidelity on one CICIDS-2017 day (Comment 12 plan §3, decision D5).

  python c12_e0.py --repo <repo> --pcap Friday-WorkingHours.pcap --labelled <dir with the day's
                   TrafficLabelling CSVs> --work <dir> --out <dir> [--max-packets N]

1. Extract flows from the PCAP with c12_flowmeter, once per fin_mode (first_fin, none).
2. Match every published flow (TrafficLabelling CSV: Flow ID columns + Timestamp + label) to an
   extracted flow: same directional 5-tuple, nearest start time after a global clock offset
   (whole hours, searched; the CSV clock may be 12-hour without AM/PM, both are tried),
   one-to-one.
3. Per extractor variant (fin_mode x flags count/binary x IAT std sample/pop), agreement of every
   column the feature mapper uses (|a-b| <= 1e-6 * max(1,|a|)), median / p99 relative error.
4. For the best variant: both feature versions (published vs re-extracted) of the matched flows are
   mapped with the frozen v2 preprocessing and scored by the deployed ML decision (no rule detectors:
   the CSV has no evasion statistics). Reports alert / class agreement and recall / FPR on both.
   NOTE: Friday rows are partly in the v2 training split, so absolute accuracies are optimistic;
   the comparison between the two versions is what E0 measures.
Writes e0.json and e0_extractor_config.json (the chosen variant, used by E1).
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
from c12_common import MAPPER_COLS, Scorer, apply_variant, rate, say, to_x33  # noqa: E402
from c12_flowmeter import Config, extract  # noqa: E402

CMP_COLS = [c for c in MAPPER_COLS if c != "Destination Port"] + ["Total Backward Packets", "Total Length of Bwd Packets"]


def load_published(d: Path, pattern: str) -> pd.DataFrame:
    frames = []
    files = sorted(d.glob(pattern))
    if not files:
        raise SystemExit(f"no published CSV matching {pattern} in {d}")
    for p in files:
        df = pd.read_csv(p, encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(subset=["Source IP", "Timestamp"])
        df["csv_file"] = p.name
        frames.append(df)
        say(f"  {p.name}: {len(df):,} rows")
    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["Timestamp"].astype(str).str.strip(), format="mixed", dayfirst=False, errors="coerce")
    df["csv_s"] = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds()
    df = df.dropna(subset=["csv_s"])
    for c in ("Source Port", "Destination Port", "Protocol"):
        df[c] = df[c].astype(float).astype(np.int64)
    df["key"] = (df["Source IP"].astype(str).str.strip() + "|" + df["Destination IP"].astype(str).str.strip() + "|"
                 + df["Source Port"].astype(str) + "|" + df["Destination Port"].astype(str) + "|" + df["Protocol"].astype(str))
    for c in CMP_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.inf)
    return df.reset_index(drop=True)


def ours_key(df):
    return (df["Source IP"] + "|" + df["Destination IP"] + "|" + df["Source Port"].astype(str) + "|"
            + df["Destination Port"].astype(str) + "|" + df["Protocol"].astype(str))


def find_offset(pub: pd.DataFrame, ours: pd.DataFrame) -> dict:
    """Whole-hour clock offset maximising unique-key matches with CSV time in [t-62 s, t+2 s]."""
    pu = pub[~pub.key.duplicated(keep=False)][["key", "csv_s"]]
    ou = ours[~ours.key.duplicated(keep=False)][["key", "start_us"]]
    m = pu.merge(ou, on="key")
    d = m.start_us.to_numpy() / 1e6 - m.csv_s.to_numpy()
    best = None
    for k in sorted(range(-26, 27), key=abs):                   # ties -> the smallest shift
        e = d - k * 3600
        ok = ((e >= -2) & (e <= 62)) | ((e - 43200 >= -2) & (e - 43200 <= 62)) | ((e + 43200 >= -2) & (e + 43200 <= 62))
        c = int(ok.sum())
        if best is None or c > best[1]:
            best = (k, c)
    return {"offset_hours": best[0], "unique_key_pairs": int(len(m)), "unique_key_pairs_in_window": best[1]}


def match(pub: pd.DataFrame, ours: pd.DataFrame, off_h: int) -> pd.DataFrame:
    """One-to-one nearest match by key; returns pairs (pub_idx, our_idx, dt_s)."""
    o = ours[["key", "start_us"]].copy(); o["our_idx"] = o.index; o["t"] = o.start_us / 1e6
    o = o.sort_values("t")
    best = None
    for shift in (0, 43200, -43200):
        p = pub[["key", "csv_s"]].copy(); p["pub_idx"] = p.index
        p["t"] = p.csv_s + off_h * 3600 + shift + 30.0                 # CSV time is truncated: flow start in [csv, csv+60)
        p = p.sort_values("t")
        m = pd.merge_asof(p, o[["key", "t", "our_idx"]].rename(columns={"t": "t_our"}), left_on="t", right_on="t_our",
                          by="key", direction="nearest", tolerance=32.0)
        m["dt"] = (m.t_our - m.t).abs()
        m = m.dropna(subset=["our_idx"])[["pub_idx", "our_idx", "dt"]]
        best = m if best is None else pd.concat([best, m])
    best = best.sort_values("dt").drop_duplicates("pub_idx").drop_duplicates("our_idx")
    return best.astype({"pub_idx": np.int64, "our_idx": np.int64})


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(float); b = b.astype(float)
    both_inf = np.isinf(a) & np.isinf(b); both_nan = np.isnan(a) & np.isnan(b)
    fin = np.isfinite(a) & np.isfinite(b)
    eq = both_inf | both_nan | (fin & (np.abs(a - b) <= 1e-6 * np.maximum(1.0, np.abs(a))))
    rel = np.abs(a[fin] - b[fin]) / np.maximum(1.0, np.abs(a[fin]))
    # the published CSV prints some columns with ~3 significant digits (e.g. Flow IAT Max 9.03E7)
    eq3 = both_inf | both_nan | (fin & (np.abs(a - b) <= 5e-3 * np.maximum(1.0, np.abs(a))))
    return {"agree_pct": 100 * float(eq.mean()), "agree_3sig_pct": 100 * float(eq3.mean()), "median_rel_err": float(np.median(rel)) if rel.size else None,
            "p99_rel_err": float(np.percentile(rel, 99)) if rel.size else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--pcap", required=True)
    ap.add_argument("--labelled", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--max-packets", type=int)
    ap.add_argument("--fin-modes", default="first_fin,fin_existing")
    ap.add_argument("--labelled-glob", default="Friday*.csv")
    a = ap.parse_args()
    repo = Path(a.repo).resolve(); sys.path.insert(0, str(repo))
    work, out = Path(a.work), Path(a.out); work.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    res = {"repo_commit": os.popen(f"git -C {repo} rev-parse HEAD").read().strip(), "pcap": str(a.pcap),
           "pcap_bytes": Path(a.pcap).stat().st_size, "max_packets": a.max_packets}

    say("published flows (TrafficLabelling CSVs)")
    pub = load_published(Path(a.labelled), a.labelled_glob)
    res["published_rows"] = int(len(pub)); res["published_labels"] = pub["Label"].value_counts().to_dict()

    extracted = {}
    for fm in a.fin_modes.split(","):
        cache = work / f"e0c_flows_{fm}{'_max' + str(a.max_packets) if a.max_packets else ''}.pkl"
        if cache.exists():
            df = pd.read_pickle(cache); st = json.loads(cache.with_suffix(".json").read_text())
            say(f"extraction fin_mode={fm}: cached ({len(df):,} flows)")
        else:
            say(f"extraction fin_mode={fm}")
            t0 = time.time()
            df, s = extract(Path(a.pcap), Config(fin_mode=fm), a.max_packets,
                            progress=lambda s, n_act, n_done: say(f"    {s.packets:,} packets, {n_act:,} active, {n_done:,} done"))
            st = {"packets": s.packets, "used": s.used, "skipped": dict(s.skipped), "seconds": round(time.time() - t0, 1),
                  "first_us": s.first_us, "last_us": s.last_us}
            df.to_pickle(cache); cache.with_suffix(".json").write_text(json.dumps(st))
        df["key"] = ours_key(df)
        say(f"    {len(df):,} flows; {st}")
        extracted[fm] = (df, st)

    res["variants"] = {}
    best = None
    for fm, (ours, st) in extracted.items():
        lo, hi = st["first_us"] / 1e6, st["last_us"] / 1e6
        off = find_offset(pub, ours)
        say(f"fin_mode={fm}: clock offset {off}")
        p = pub.copy()
        # only published flows inside the extracted time span (relevant for --max-packets smoke runs)
        t_adj = p.csv_s + off["offset_hours"] * 3600
        inside = ((t_adj >= lo - 62) & (t_adj <= hi)) | ((t_adj + 43200 >= lo - 62) & (t_adj + 43200 <= hi)) | \
                 ((t_adj - 43200 >= lo - 62) & (t_adj - 43200 <= hi))
        p = p[inside]
        m = match(p, ours, off["offset_hours"])
        entry = {"extraction": st, "extracted_flows": int(len(ours)), "clock": off, "published_in_span": int(len(p)),
                 "matched": int(len(m)), "matched_pct_of_published": 100 * len(m) / max(len(p), 1),
                 "matched_pct_of_extracted": 100 * len(m) / max(len(ours), 1),
                 "unmatched_published_by_label": p.drop(index=m.pub_idx)["Label"].value_counts().head(20).to_dict(),
                 "median_dt_s": float(m.dt.median()) if len(m) else None}
        P = pub.loc[m.pub_idx.to_numpy()].reset_index(drop=True)
        base = ours.loc[m.our_idx.to_numpy()].reset_index(drop=True)
        for flags, std, pay in [(f, s, p) for f in ("count", "binary", "cicv3", "cicv3b") for s in ("sample", "pop") for p in ("ip", "frame")]:
            if True:
                O = apply_variant(base, {"flags": flags, "iat_std": std, "payload": pay})
                cols = {c: agreement(P[c].to_numpy(), O[c].to_numpy()) for c in CMP_COLS}
                mean_agree = float(np.mean([v["agree_pct"] for v in cols.values()]))
                name = f"{fm}/{flags}/{std}/{pay}"
                entry[name] = {"mean_agree_pct": mean_agree, "columns": cols}
                say(f"    {name}: mean column agreement {mean_agree:.2f} %")
                score = mean_agree * len(m) / max(len(p), 1)        # agreement weighted by match coverage
                if best is None or score > best[0]:
                    best = (score, {"fin_mode": fm, "flags": flags, "iat_std": std, "payload": pay}, m, P, O)
        res["variants"][fm] = entry

    _, cfg, m, P, O = best
    res["chosen_variant"] = cfg
    say(f"chosen variant {cfg}; scoring {len(m):,} matched flows twice (published vs re-extracted)")
    from edge.training.label_mapper import CLASS_NAMES, map_single_label
    sc = Scorer()
    res["policy_version"] = sc.policy.version; res["model_binding"] = sc.binding
    y = np.array([CLASS_NAMES[map_single_label(str(v))] for v in P["Label"]])
    dport = P["Destination Port"].to_numpy(); proto = P["Protocol"].to_numpy()
    Xp, Xo = to_x33(P, repo), to_x33(O, repo)
    res["ingestor_effect"] = sc.ingestor_effect(Xo, dport, proto)
    rp, ro = sc.score(Xp, dport, proto), sc.score(Xo, dport, proto)
    atk = y != "BENIGN"
    ml_p, ml_o = rp.ml_alert.to_numpy(), ro.ml_alert.to_numpy()
    res["model_comparison"] = {
        "flows": int(len(y)), "attack_flows": int(atk.sum()),
        "ml_alert_agreement_pct": 100 * float((ml_p == ml_o).mean()),
        "ml_top_class_agreement_pct": 100 * float((rp.ml_top.to_numpy() == ro.ml_top.to_numpy()).mean()),
        "published": {"recall": rate(ml_p[atk].sum(), atk.sum()), "fpr": rate(ml_p[~atk].sum(), (~atk).sum()),
                      "naming_on_detected": rate((rp.ml_top.to_numpy()[atk & ml_p] == y[atk & ml_p]).sum(), (atk & ml_p).sum())},
        "re_extracted": {"recall": rate(ml_o[atk].sum(), atk.sum()), "fpr": rate(ml_o[~atk].sum(), (~atk).sum()),
                         "naming_on_detected": rate((ro.ml_top.to_numpy()[atk & ml_o] == y[atk & ml_o]).sum(), (atk & ml_o).sum())},
        "per_label": {lab: {"n": int((y == lab).sum()), "published_alert_pct": 100 * float(ml_p[y == lab].mean()),
                            "re_extracted_alert_pct": 100 * float(ml_o[y == lab].mean())} for lab in np.unique(y)},
        "note": "ML decision only (s_attack >= tau); Friday rows are partly in the v2 training split",
    }
    say(f"model comparison: {json.dumps({k: v for k, v in res['model_comparison'].items() if k != 'per_label'})}")
    (out / "e0.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "e0_extractor_config.json").write_text(json.dumps(
        {**cfg, "timeout_s": 120.0, "chosen_by": "c12_e0.py", "repo_commit": res["repo_commit"]}, indent=1))
    say("written e0.json, e0_extractor_config.json")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
