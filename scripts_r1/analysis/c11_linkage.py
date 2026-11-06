"""
C11 — Comment 9 evidence 4.2: quasi-identifier linkage risk of what the edge relays.

  python c11_linkage.py --repo <repo> --cache <dir with prepared_*.npz> --out <dir> [--x2 <x2_x3.json>]

On the 2,431,994 deduplicated v2 flows:
 (a) uniqueness / k-anonymity of the relayed feature vector: 28 cloud-safe fields (layout alert35/1,
     before Comment 9) vs the 23 payload fields (alert30/2), and of coarse quasi-identifiers
     (port class + log2-binned bytes / packets / duration), with and without the exact port number;
 (b) can the removed PHI-adjacent flag is_FHIR (443/8080) be inferred from the 23 relayed features?
     (is_DICOM / is_HL7 occur on 1 / 7 rows only in CICIDS-2017 and cannot be evaluated.) Before D3 it
     was exactly recoverable from dst_port_number; after D3 a classifier is trained and evaluated;
 (c) aggregate linkability: share of 250 ms windows that contain exactly one alert (an "aggregate"
     equal to one alert), analytically (Poisson) and from the X2 runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(repo, cache):
    committed = np.load(repo / "edge/models_v2/split_indices_v2.npz")
    for c in sorted(Path(cache).glob("prepared_*.npz")):
        z = np.load(c)
        if all(np.array_equal(z[f"part_{k}"], committed[k]) for k in ("train", "val", "tune", "test")):
            return z["X"].astype(np.float32), z["y"]
    raise SystemExit("no cache matches the committed v2 split")


def kanon(df: pd.DataFrame) -> dict:
    """Equivalence-class size of each row (exact match on all columns), via 64-bit row hashes."""
    h = pd.util.hash_pandas_object(df, index=False)
    counts = h.map(h.value_counts()).to_numpy()
    return {"rows": int(len(df)), "equivalence_classes": int(h.nunique()),
            "unique_rows_pct": 100 * float((counts == 1).mean()),
            "k_le_5_pct": 100 * float((counts <= 5).mean()), "k_le_10_pct": 100 * float((counts <= 10).mean()),
            "median_k": float(np.median(counts))}


def lg2(v):
    return np.floor(np.log2(np.maximum(v, 0) + 1)).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--x2")
    a = ap.parse_args()
    repo = Path(a.repo).resolve(); sys.path.insert(0, str(repo))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from contracts.flow_feature_vector import CLOUD_SAFE_INDICES, FEATURE_NAMES
    from edge.crypto.alert_encryption import PAYLOAD_FEATURE_INDICES
    X, y = load(repo, a.cache)
    say(f"{len(X):,} flows")
    res = {"n_flows": int(len(X)), "repo_commit": os.popen(f"git -C {repo} rev-parse HEAD").read().strip()}

    # (a) uniqueness
    cols28, cols23 = list(CLOUD_SAFE_INDICES), list(PAYLOAD_FEATURE_INDICES)
    port = np.round(X[:, 31] * 65535).astype(np.int32)
    coarse = pd.DataFrame({"well_known": X[:, 15].astype(np.int8), "ephemeral": X[:, 16].astype(np.int8),
                           "bytes_l2": lg2(X[:, 0]), "pkts_l2": lg2(X[:, 1]), "dur_l2": lg2(X[:, 4])})
    res["uniqueness"] = {}
    for tag, df in (("alert35_1_28_fields", pd.DataFrame(X[:, cols28])),
                    ("alert30_2_23_fields", pd.DataFrame(X[:, cols23])),
                    ("coarse_qi_portclass_bytes_pkts_duration", coarse),
                    ("coarse_qi_plus_exact_port", coarse.assign(port=port))):
        say(f"(a) {tag}"); res["uniqueness"][tag] = kanon(df); say(f"    {res['uniqueness'][tag]}")
    attacks = y != 0
    res["uniqueness_alerting_class_rows_only"] = {
        "alert30_2_23_fields": kanon(pd.DataFrame(X[attacks][:, cols23]))}

    # (b) is_FHIR inference from the relayed fields
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    t = X[:, 22] > 0.5
    rng = np.random.default_rng(0); idx = rng.permutation(len(X))
    tr, te = idx[: len(X) // 2], idx[len(X) // 2:]
    feats = X[:, cols23]
    m = lgb.LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.1, n_jobs=-1, verbose=-1, random_state=0)
    say("(b) training is_FHIR inference model on the 23 relayed features")
    m.fit(feats[tr], t[tr])
    p = m.predict_proba(feats[te])[:, 1]
    pred = p >= 0.5
    imp = sorted(zip(m.booster_.feature_importance("gain"), [FEATURE_NAMES[i] for i in cols23]), reverse=True)[:5]
    res["is_FHIR_inference"] = {
        "prevalence_pct": 100 * float(t.mean()), "rows_is_DICOM": int((X[:, 20] > 0.5).sum()),
        "rows_is_HL7": int((X[:, 21] > 0.5).sum()),
        "before_D3": "exact (is_FHIR = dst_port in {443, 8080}; dst_port_number was relayed)",
        "after_D3_auroc": float(roc_auc_score(t[te], p)),
        "after_D3_accuracy": float((pred == t[te]).mean()),
        "after_D3_majority_baseline_accuracy": float(max(t[te].mean(), 1 - t[te].mean())),
        "after_D3_recall_pct": 100 * float(pred[t[te]].mean()), "after_D3_precision_pct":
            100 * float(t[te][pred].mean()) if pred.any() else None,
        "top_features_by_gain": [(n, float(g)) for g, n in imp]}
    say(f"    {res['is_FHIR_inference']}")

    # (c) aggregate linkability
    W = 0.25
    def p_single(rate):
        lam = rate * W
        return lam * math.exp(-lam) / (1 - math.exp(-lam)) if lam > 0 else 1.0
    res["aggregate_linkability"] = {
        "window_s": W, "definition": "P(window holds exactly 1 alert | it holds >= 1), Poisson arrivals",
        "by_alert_rate_per_s": {str(r): p_single(r) for r in (0.1, 1, 4, 10, 18, 50, 100)}}
    if a.x2 and Path(a.x2).exists():
        x2 = json.loads(Path(a.x2).read_text()); obs = []
        for v in x2.values():
            if isinstance(v, dict) and "ramp" in v:
                for s in v["ramp"] + [v[k] for k in ("busy", "burst") if k in v]:
                    if s.get("batches"):
                        obs.append({"label": s["label"], "offered": s["offered_rate"],
                                    "alerts_per_s": s["alerts_encrypt_selected"] / s["duration_s"],
                                    "alerts_per_ciphertext": s["alerts_per_batch"]})
        res["aggregate_linkability"]["x2_observed"] = obs
    say(f"(c) {res['aggregate_linkability']['by_alert_rate_per_s']}")
    (out / "c11_linkage.json").write_text(json.dumps(res, indent=1)); say("written c11_linkage.json")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
