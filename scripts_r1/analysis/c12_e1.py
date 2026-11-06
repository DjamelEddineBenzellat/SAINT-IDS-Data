"""
C12 E1 — cross-dataset transfer on CICIoMT2024 (Wi-Fi / MQTT PCAPs), v2 model and tau = 0.25 UNCHANGED.
Runs only after the inventory, with a label map fixed before scoring (plan §4).

  python c12_e1.py extract --root <CICIoMT2024 dir> --label-map label_map.json \
                           --extractor e0_extractor_config.json --work <dir> [--jobs 4]
  python c12_e1.py score   --repo <repo> --label-map label_map.json --extractor e0_extractor_config.json \
                           --work <dir> --out <dir> [--parity 1000]

label_map.json:
  {"include_globs": ["WiFI_and_MQTT/attacks/PCAP/test/*", ...],   (optional: only these captures are used)
   "exclude_globs": ["*Bluetooth*"],
   "files": [{"glob": "*Benign*", "label": "BENIGN"},
             {"glob": "*TCP_IP-DDoS-SYN*", "label": "DDoS-SYN", "saint_class": "DOS",
              "attacker_ips": ["192.168.x.y"], "other_flows": "exclude"}, ...]}
  - first matching glob wins; a capture matching no entry is an error (nothing is labelled silently);
  - attacker_ips: flows with the attacker as source or destination get the attack label; other flows in
    that capture are "benign" or "exclude" (other_flows); attacker_ips null = every flow gets the label;
  - saint_class null = no SAINT class (detection only, excluded from naming).
Every flow goes through the deployed decision path (c12_common.Scorer: ingestor -> ONNX -> rules -> decide()),
with the 6 evasion statistics. A random --parity sample also goes through the full API handler
(Handlers._scan_single with native SHAP and the AlertRouter, temp logs) and must give identical decisions.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c12_common import Scorer, apply_variant, detection_naming, rate, say, to_x33  # noqa: E402
from c12_flowmeter import EVASION_COLS, Config, extract  # noqa: E402

CAPTURE = {".pcap", ".pcapng", ".cap"}


def rule_for(rel: str, lm: dict):
    rl = rel.lower()                                   # all globs are case-insensitive
    if any(fnmatch.fnmatch(rl, g.lower()) for g in lm.get("exclude_globs", [])):
        return "exclude"
    for r in lm["files"]:
        if fnmatch.fnmatch(rl, r["glob"].lower()):
            return r
    return None


def _active_seconds(p: Path, max_gap_s: float = 300.0) -> float:
    """Capture time that actually contains traffic: sum of inter-packet gaps <= max_gap_s (some profiling
    captures span days with long pauses, so last - first would inflate 'false alarms per hour')."""
    from c12_flowmeter import open_reader
    fh, rd, _ = open_reader(p); tot, prev = 0.0, None
    try:
        for ts, _buf in rd:
            if prev is not None and 0 <= ts - prev <= max_gap_s:
                tot += ts - prev
            prev = ts
    finally:
        fh.close()
    return tot


def _extract_one(args):
    p, rel, cfg, work = args
    dst = work / (rel.replace("/", "__") + ".pkl")
    if dst.exists():
        return rel, "cached"
    t0 = time.time()
    df, st = extract(p, cfg, progress=lambda s, n_act, n_done: say(f"    {rel}: {s.packets:,} packets, {n_act:,} active flows")
                     if s.packets % 5_000_000 == 0 else None)
    active_s = _active_seconds(p)
    df["capture"] = rel
    df.to_pickle(dst)
    (dst.with_suffix(".json")).write_text(json.dumps({"packets": st.packets, "used": st.used, "skipped": dict(st.skipped),
                                                     "first_us": st.first_us, "last_us": st.last_us, "active_s": active_s,
                                                     "flows": len(df), "flows_total": st.flows_total,
                                                     "sample_frac": len(df) / st.flows_total if st.flows_total else 1.0,
                                                     "peak_active": st.peak_active, "seconds": round(time.time() - t0, 1)}))
    say(f"  {rel}: {len(df):,} flows kept of {st.flows_total:,} from {st.packets:,} packets ({time.time() - t0:.0f} s)")
    return rel, "done"


def cmd_extract(a):
    root = Path(a.root).resolve(); work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    lm = json.loads(Path(a.label_map).read_text()); ex = json.loads(Path(a.extractor).read_text())
    cfg = Config(timeout_s=ex.get("timeout_s", 120.0), fin_mode=ex["fin_mode"], include_icmp=True,
                 max_flows=a.max_flows or None, seed=0)
    jobs, missing = [], []
    for p in sorted(q for q in root.rglob("*") if q.suffix.lower() in CAPTURE):
        rel = str(p.relative_to(root))
        if lm.get("include_globs") and not any(fnmatch.fnmatch(rel.lower(), g.lower()) for g in lm["include_globs"]):
            continue
        r = rule_for(rel, lm)
        if r == "exclude":
            continue
        if r is None:
            missing.append(rel); continue
        jobs.append((p, rel, cfg, work))
    if missing:
        raise SystemExit(f"captures without a label-map entry (fix the map first): {missing[:20]}")
    say(f"extracting {len(jobs)} captures with {cfg} ({a.jobs} jobs)")
    with Pool(a.jobs) as pool:
        pool.map(_extract_one, jobs, chunksize=1)


def labelled(work: Path, lm: dict, pkls=None) -> pd.DataFrame:
    """Label the cached flow tables (all of `work`, or only `pkls`); empty frame if nothing is kept."""
    frames = []
    for pkl in (sorted(work.glob("*.pkl")) if pkls is None else pkls):
        df = pd.read_pickle(pkl); rel = df["capture"].iloc[0] if len(df) else pkl.stem
        r = rule_for(rel, lm)
        if r in (None, "exclude"):
            continue
        att = r.get("attacker_ips")
        if r.get("attackers"):
            # attack flow = attacker <-> anyone in the attack protocol / on the attack port, in EITHER direction
            # (E1 diag 2026-10-30: in several captures most attack flows start with the victim's packet, e.g. after
            # the 120 s timeout with port reuse; direction "initiator" keeps only attacker-first flows, as in run 1);
            # other flows touching an attacker are ambiguous -> excluded; flows not touching one -> other_flows
            A = set(r["attackers"])
            src, dst = df["Source IP"].isin(A), df["Destination IP"].isin(A)
            proto = df["Protocol"].isin(r["protocols"]) if r.get("protocols") else pd.Series(True, index=df.index)
            pf = df["Destination Port"].isin(r["dst_ports"]) if r.get("dst_ports") else pd.Series(True, index=df.index)
            pr = df["Source Port"].isin(r["dst_ports"]) if r.get("dst_ports") else pd.Series(True, index=df.index)
            m = src & proto & pf
            if r.get("direction", lm.get("direction", "either")) == "either":
                m |= dst & ~src & proto & pr
            keep = m | (~src & ~dst & (r.get("other_flows", "exclude") == "benign"))
            df = df[keep].copy(); m = m[keep]
            df["is_attack"] = m.to_numpy()
            df["label"] = np.where(df.is_attack, r["label"], "BENIGN")
            df["true_class"] = np.where(df.is_attack, r.get("saint_class") or "", "BENIGN")
            df.loc[df.true_class == "", "true_class"] = None
        elif r["label"] == "BENIGN":
            df["label"], df["true_class"], df["is_attack"] = "BENIGN", "BENIGN", False
        elif att is None:
            df["label"], df["true_class"], df["is_attack"] = r["label"], r.get("saint_class"), True
        else:
            hit = df["Source IP"].isin(att) | df["Destination IP"].isin(att)
            if r.get("other_flows", "exclude") == "exclude":
                df = df[hit].copy(); hit = hit[hit]
            df["is_attack"] = hit.to_numpy()
            df["label"] = np.where(df.is_attack, r["label"], "BENIGN")
            df["true_class"] = np.where(df.is_attack, r.get("saint_class") or "", "BENIGN")
            df.loc[df.true_class == "", "true_class"] = None
        meta = json.loads(pkl.with_suffix(".json").read_text())
        df["capture_seconds"] = meta.get("active_s", ((meta["last_us"] or 0) - (meta["first_us"] or 0)) / 1e6)
        df["weight"] = 1.0 / meta.get("sample_frac", 1.0)          # flows represented by each kept flow
        df["flows_total_capture"] = meta.get("flows_total", len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def episodes(df: pd.DataFrame) -> dict:
    """One attack run = one capture file: detected if >= 1 attack flow alerts; time to first alert."""
    out = {}
    for cap, g in df[df.is_attack].groupby("capture"):
        g = g.sort_values("start_us"); t0 = g.start_us.iloc[0]
        al = g[g.alert]
        out[cap] = {"label": g.label.iloc[0], "attack_flows": int(len(g)), "alerted_flows": int(len(al)),
                    "detected": bool(len(al)),
                    "time_to_first_alert_s": float((al.start_us.iloc[0] + al["Flow Duration"].iloc[0] - t0) / 1e6) if len(al) else None}
    return out


def parity(sc: Scorer, df: pd.DataFrame, X: np.ndarray, n: int) -> dict:
    from edge.api.handlers import Handlers
    from edge.api.schemas import PacketScan
    from edge.crypto.pseudonym import IPPseudonymizer
    from edge.inference.alert_router import AlertRouter
    from edge.inference.decision import serving_models_dir
    from edge.inference.explainer import SHAPExplainer
    h = Handlers(flow_ingestor=sc.ing, scoring_engine=sc.se,
                 explainer=SHAPExplainer(models_dir=serving_models_dir(sc.policy), policy=sc.policy),
                 alert_router=AlertRouter(log_dir=tempfile.mkdtemp(), max_alerts_per_minute=10**9,
                                          pseudonymizer=IPPseudonymizer(key="c12-parity")))
    idx = np.random.default_rng(1).choice(len(df), min(n, len(df)), replace=False)
    bad, expl = 0, {}
    for i in idx:
        r = df.iloc[i]
        meta = sc.metadata(X[i], r["Destination Port"], r["Protocol"], {k: r[k] for k in EVASION_COLS})
        resp = h._scan_single(PacketScan(metadata=meta))
        expl[resp.explanation_status] = expl.get(resp.explanation_status, 0) + 1
        if (resp.severity != "INFO") != bool(r.alert) or resp.predicted_class != r.pred or "|".join(resp.alert_sources) != r.sources:
            bad += 1
    return {"n": int(len(idx)), "mismatches": bad, "explanation_status": expl}


def cmd_score(a):
    repo = Path(a.repo).resolve(); sys.path.insert(0, str(repo))
    work, out = Path(a.work), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lm = json.loads(Path(a.label_map).read_text()); ex = json.loads(Path(a.extractor).read_text())
    df = apply_variant(labelled(work, lm), ex)
    say(f"{len(df):,} labelled flows; labels: {df.label.value_counts().to_dict()}")
    X = to_x33(df, repo)
    sc = Scorer()
    dport, proto = df["Destination Port"].to_numpy(), df["Protocol"].to_numpy()
    res = {"repo_commit": os.popen(f"git -C {repo} rev-parse HEAD").read().strip(), "policy_version": sc.policy.version,
           "model_binding": sc.binding, "extractor": ex, "label_map": lm, "flows": int(len(df)),
           "ingestor_effect": sc.ingestor_effect(X, dport, proto)}
    r = sc.score(X, dport, proto, evasion=df[list(EVASION_COLS)])
    for c in r.columns:
        df[c] = r[c].to_numpy()
    df["dport"] = dport
    res["all"] = detection_naming(df)
    benign_caps = [c for c, g in df.groupby("capture") if not g.is_attack.any() and (g.label == "BENIGN").all()
                   and rule_for(c, lm) not in (None, "exclude") and rule_for(c, lm)["label"] == "BENIGN"]
    bo = df[df.capture.isin(benign_caps)]
    hours = float(bo.groupby("capture").capture_seconds.first().sum() / 3600) if len(bo) else 0.0
    wa, wm = float((bo.weight * bo.alert).sum()), float((bo.weight * bo.ml_alert).sum())
    res["benign_only_captures"] = {"captures": benign_caps, "flows": int(len(bo)), "flows_represented": float(bo.weight.sum()),
                                   "active_hours": hours,
                                   "false_alarm_rate": rate(bo.alert.sum(), len(bo)),
                                   "false_alarms_per_hour": wa / hours if hours else None,
                                   "ml_false_alarms_per_hour": wm / hours if hours else None,
                                   "rule_false_alarms_per_hour": (wa - wm) / hours if hours else None}
    atk = df[df.is_attack]
    res["weighted_pooled"] = {
        "note": "each kept flow weighted by 1/sample_frac of its capture (reservoir sampling for large captures)",
        "attack_detection": float((atk.weight * atk.alert).sum() / atk.weight.sum()) if len(atk) else None,
        "attack_detection_by_ml": float((atk.weight * atk.ml_alert).sum() / atk.weight.sum()) if len(atk) else None,
        "attack_flows_represented": float(atk.weight.sum())}
    res["per_capture"] = {}
    for cap, g in df.groupby("capture"):
        a_, b_ = g[g.is_attack], g[~g.is_attack]
        res["per_capture"][cap] = {
            "label": str(a_.label.iloc[0]) if len(a_) else "BENIGN", "flows_total": int(g.flows_total_capture.iloc[0]),
            "kept_after_labelling": int(len(g)), "weight": float(g.weight.iloc[0]), "active_s": float(g.capture_seconds.iloc[0]),
            "attack_flows": int(len(a_)), "attack_alert_pct": 100 * float(a_.alert.mean()) if len(a_) else None,
            "attack_ml_alert_pct": 100 * float(a_.ml_alert.mean()) if len(a_) else None,
            "attack_naming_pct": 100 * float((a_[a_.alert].pred == a_[a_.alert].true_class).mean())
            if len(a_) and a_.alert.any() and a_.true_class.notna().any() else None,
            "attack_pred_classes": a_[a_.alert].pred.value_counts().head(5).to_dict(),
            "benign_flows": int(len(b_)), "benign_alerts": int(b_.alert.sum()), "benign_ml_alerts": int(b_.ml_alert.sum()),
            "alert_sources": g[g.alert].sources.value_counts().head(6).to_dict()}
    res["ml_only_view"] = {"attack_detection_by_ml": rate(df[df.is_attack].ml_alert.sum(), df.is_attack.sum()),
                           "benign_ml_alert_rate": rate(df[~df.is_attack].ml_alert.sum(), (~df.is_attack).sum())}
    # alternative alerting views from the recorded alert sources (no re-scoring): which component alerts
    ev = df.sources.fillna("").str.contains("rule:evasion")
    ff = df.sources.fillna("").str.contains("rule:fast_filter")
    views = {"full_policy_2.3.0": df.alert.to_numpy(), "ml_only": df.ml_alert.to_numpy(),
             "ml_plus_fast_filter (evasion rules off)": (df.ml_alert | ff).to_numpy(), "evasion_rules_only": ev.to_numpy()}
    res["views"] = {}
    bmask = (~df.is_attack).to_numpy()
    for name, al in views.items():
        d = {"benign_alert_rate": rate(al[bmask].sum(), bmask.sum()),
             "benign_only_captures_alerts_per_hour": float((bo.weight.to_numpy() * al[df.capture.isin(benign_caps).to_numpy()]).sum() / hours) if hours else None,
             "attack_detection_weighted": float((atk.weight.to_numpy() * al[df.is_attack.to_numpy()]).sum() / atk.weight.sum()) if len(atk) else None,
             "per_label_detection_pct": {}}
        for lab, g in df[df.is_attack].groupby("label"):
            d["per_label_detection_pct"][lab] = 100 * float(al[g.index.to_numpy()].mean())
        res["views"][name] = d
    # naming by the MODEL (ml_top among ML alerts); `pred` is the rule's class whenever a rule fires
    res["naming_ml"] = {}
    for lab, g in df[df.is_attack].groupby("label"):
        ml = g[g.ml_alert]
        tc = g.true_class.iloc[0]
        res["naming_ml"][lab] = {"true_class": tc, "ml_alerts": int(len(ml)), "flows": int(len(g)),
                                 "ml_top": ml.ml_top.value_counts().to_dict(),
                                 "ml_naming": rate((ml.ml_top == tc).sum(), len(ml)) if len(ml) and isinstance(tc, str) else None}
    res["episodes"] = episodes(df)
    res["s_attack_quantiles"] = {str(k): {q: float(g.s_attack.quantile(q)) for q in (0.5, 0.9, 0.99)}
                                 for k, g in df.groupby("label")}
    if a.parity:
        say(f"parity: {a.parity} random flows through the full API handler (SHAP + router)")
        res["parity_full_handler"] = parity(sc, df, X, a.parity)
    (out / "e1.json").write_text(json.dumps(res, indent=1, default=str))
    df[["capture", "label", "true_class", "is_attack", "Destination Port", "Protocol", "alert", "ml_alert", "s_attack",
        "pred", "ml_top", "sources"]].to_csv(out / "e1_decisions.csv.gz", index=False)
    say(f"written e1.json; detection {res['all']['attack']['detection']}, benign {res['all']['benign']['false_alarm_rate']}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--root", required=True); e.add_argument("--label-map", required=True)
    e.add_argument("--extractor", required=True); e.add_argument("--work", required=True); e.add_argument("--jobs", type=int, default=1)
    e.add_argument("--max-flows", type=int, default=400_000, help="uniform reservoir sample per capture (0 = all)")
    s = sub.add_parser("score"); s.add_argument("--repo", required=True); s.add_argument("--label-map", required=True)
    s.add_argument("--extractor", required=True); s.add_argument("--work", required=True); s.add_argument("--out", required=True)
    s.add_argument("--parity", type=int, default=1000)
    a = ap.parse_args()
    cmd_extract(a) if a.cmd == "extract" else cmd_score(a)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
