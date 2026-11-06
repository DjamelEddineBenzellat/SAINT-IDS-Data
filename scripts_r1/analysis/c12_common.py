"""
C12 — shared helpers for Comment 12 (E0, E1): feature mapping with the frozen v2 preprocessing,
scoring through the deployed components, detection / naming metrics.

Mapping (identical to how the v2 model's training data was built):
  extractor columns (CICFlowMeter names) -> edge.training.feature_mapper.map_cicids_to_features,
  WITHOUT 'Protocol' / 'Timestamp' columns (the CICIDS-2017 CSVs used for training had neither),
  then +/-inf cells of Flow Packets/s (idx 3) and Flow Bytes/s (idx 23) set to the v2 TRAIN-ONLY
  values recorded in edge/models_v2/training_report.json (data.inf_replacement).
Scoring: FlowIngestor.ingest(metadata) -> ScoringEngine.score (ONNX + rule detectors + decide(), the
  deployed policy file and its bound model). The metadata carries the 33 canonical features, the
  destination port, the protocol and (when available) the 6 evasion statistics.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

MAPPER_COLS = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
               "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
               "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "ACK Flag Count", "Init_Win_bytes_forward"]
PROTOCOL_PORTS = {"DICOM": {104, 11112}, "HL7": {2575}, "HTTPS/FHIR(443,8080)": {443, 8080},
                  "MQTT": {1883, 8883}, "CoAP": {5683}, "DNS": {53}}


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def protocol_of(port: int) -> str:
    for k, v in PROTOCOL_PORTS.items():
        if port in v:
            return k
    return "other"


def apply_variant(df: pd.DataFrame, variant: dict) -> pd.DataFrame:
    """Select extractor variant columns: flags 'count'|'binary'|'cicv3', iat_std 'sample'|'pop',
    payload 'ip'|'frame'.
    cicv3  (E0b, first_fin splitting): FIN / ACK = flag present on the flow's FIRST packet, SYN = RST = 0.
    cicv3b (E0c, fin_existing splitting): ACK = flag on the FIRST packet, FIN = SYN = RST = 0."""
    out = df.copy()
    if variant.get("flags") == "binary":
        for c in ("FIN", "SYN", "RST", "ACK"):
            out[f"{c} Flag Count"] = out[f"{c} Flag Count_bin"]
    elif variant.get("flags") in ("cicv3", "cicv3b"):
        ff = out["first_flags"].to_numpy().astype(np.int64)
        out["FIN Flag Count"] = (ff & 0x01 > 0).astype(np.int64) if variant["flags"] == "cicv3" else 0
        out["ACK Flag Count"] = (ff & 0x10 > 0).astype(np.int64)
        out["SYN Flag Count"] = 0; out["RST Flag Count"] = 0
    if variant.get("payload") == "frame":
        for c in ("Total Length of Fwd Packets", "Total Length of Bwd Packets", "Flow Bytes/s"):
            out[c] = out[f"{c}_frame"]
    if variant.get("iat_std") == "pop":
        out["Flow IAT Std"] = out["Flow IAT Std_pop"]
    return out


def v2_inf_values(repo: Path) -> dict:
    r = json.loads((repo / "edge/models_v2/training_report.json").read_text())
    return {int(k): float(v["train_only_value"]) for k, v in r["data"]["inf_replacement"].items()}


def to_x33(df: pd.DataFrame, repo: Path) -> np.ndarray:
    from edge.training.feature_mapper import map_cicids_to_features
    sub = df[MAPPER_COLS].astype(float).reset_index(drop=True)
    inf3 = np.isinf(sub["Flow Packets/s"].to_numpy()); inf23 = np.isinf(sub["Flow Bytes/s"].to_numpy())
    X = map_cicids_to_features(sub).astype(np.float32)
    vals = v2_inf_values(repo)
    X[inf3, 3] = vals[3]; X[inf23, 23] = vals[23]
    return X


class Scorer:
    """The deployed decision path (ingestor + scoring engine + rules + decide()), model binding asserted."""

    def __init__(self):
        from edge.inference.decision import get_default_policy, serving_models_dir
        from edge.inference.flow_ingestor import FlowIngestor
        from edge.inference.scoring_engine import ScoringEngine
        self.policy = get_default_policy()
        self.se = ScoringEngine(models_dir=serving_models_dir(self.policy), policy=self.policy)
        self.binding = self.se.get_model_info()["model_binding"]
        assert self.binding["match"] is True, self.binding
        self.ing = FlowIngestor()

    @staticmethod
    def metadata(x33, dport, proto, evasion=None, src=None, dst=None) -> dict:
        from contracts.flow_feature_vector import FEATURE_NAMES
        m = {n: float(v) for n, v in zip(FEATURE_NAMES, x33)}
        m.update({"Destination Port": int(dport), "Protocol": int(proto)})
        if src is not None:
            m["Src IP"], m["Dst IP"] = src, dst
        if evasion is not None:
            m.update({k: float(v) for k, v in evasion.items()})
        return m

    def score(self, X: np.ndarray, dport, proto, evasion: pd.DataFrame | None = None, every=200_000) -> pd.DataFrame:
        from contracts.flow_feature_vector import EVASION_FEATURE_NAMES
        ev = evasion[list(EVASION_FEATURE_NAMES)].to_numpy(float) if evasion is not None else None
        out = {"alert": np.zeros(len(X), bool), "ml_alert": np.zeros(len(X), bool), "s_attack": np.zeros(len(X)),
               "pred": np.empty(len(X), object), "ml_top": np.empty(len(X), object), "sources": np.empty(len(X), object)}
        from edge.training.label_mapper import CLASS_NAMES
        t0 = time.time()
        for i in range(len(X)):
            e = dict(zip(EVASION_FEATURE_NAMES, ev[i])) if ev is not None else None
            r = self.se.score(self.ing.ingest(self.metadata(X[i], dport[i], proto[i], e)))
            out["alert"][i] = r.alert; out["s_attack"][i] = r.attack_score
            out["ml_alert"][i] = "ml" in r.alert_sources
            out["pred"][i] = r.predicted_class; out["sources"][i] = "|".join(r.alert_sources)
            p = r.probabilities
            out["ml_top"][i] = max(CLASS_NAMES[1:len(p)], key=lambda c: p[c])
            if every and (i + 1) % every == 0:
                say(f"    scored {i + 1:,}/{len(X):,} ({(i + 1) / (time.time() - t0):,.0f}/s)")
        return pd.DataFrame(out)

    def ingestor_effect(self, X: np.ndarray, dport, proto, n=5000, seed=0) -> dict:
        """Max |p_ingested - p_raw|: raw = the training-identical 33-vector scored without the ingestor."""
        from datetime import datetime, timezone

        from edge.inference.flow_ingestor import FlowFeatureVector
        idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
        diffs, changed = [], 0
        for i in idx:
            a = self.se.score(self.ing.ingest(self.metadata(X[i], dport[i], proto[i])))
            v = FlowFeatureVector(features=X[i].astype(np.float32), source_ip="0.0.0.0", dest_ip="0.0.0.0",
                                  source_port=0, dest_port=int(dport[i]), protocol=int(proto[i]),
                                  timestamp=datetime.now(timezone.utc).isoformat(), raw_field_count=33)
            b = self.se.score(v)
            diffs.append(max(abs(a.probabilities[k] - b.probabilities[k]) for k in a.probabilities))
            changed += int(a.alert != b.alert)
        return {"n": int(len(idx)), "max_abs_prob_diff": float(max(diffs)), "decisions_changed": changed}


def clopper_pearson(k: int, n: int, a: float = 0.05):
    from scipy.stats import beta
    if n == 0:
        return None, None
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def rate(k, n):
    lo, hi = clopper_pearson(int(k), int(n))
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "ci95": [lo, hi]}


def detection_naming(df: pd.DataFrame, hours_benign: float | None = None) -> dict:
    """
    df columns: label (source label), true_class (SAINT class name or None for 'no SAINT class'),
    is_attack, alert, ml_alert, pred, sources, dport.
    Detection = alert yes/no; naming = pred == true_class among detected attacks with a SAINT class.
    """
    res = {}
    ben, atk = df[~df.is_attack], df[df.is_attack]
    res["benign"] = {"flows": int(len(ben)), "false_alarm_rate": rate(ben.alert.sum(), len(ben)),
                     "ml_false_alarms": int(ben.ml_alert.sum()),
                     "rule_false_alarms": int((ben.alert & ~ben.ml_alert).sum()),
                     "false_alarm_sources": ben[ben.alert].sources.value_counts().head(15).to_dict()}
    if hours_benign:
        res["benign"]["benign_hours"] = hours_benign
        res["benign"]["false_alarms_per_hour"] = float(ben.alert.sum() / hours_benign)
    res["attack"] = {"flows": int(len(atk)), "detection": rate(atk.alert.sum(), len(atk)),
                     "detected_by_ml": int(atk.ml_alert.sum()),
                     "detected_by_rules_only": int((atk.alert & ~atk.ml_alert).sum())}
    named = atk[atk.alert & atk.true_class.notna()]
    res["attack"]["naming"] = rate((named.pred == named.true_class).sum(), len(named))
    res["per_label"] = {}
    for lab, g in df.groupby("label"):
        d = {"flows": int(len(g)), "is_attack": bool(g.is_attack.iloc[0]), "alert": rate(g.alert.sum(), len(g)),
             "ml_alert": int(g.ml_alert.sum()), "true_class": g.true_class.iloc[0]}
        nm = g[g.alert & g.true_class.notna()] if g.is_attack.iloc[0] else g.iloc[:0]
        if len(nm):
            d["naming"] = rate((nm.pred == nm.true_class).sum(), len(nm))
            d["predicted_classes"] = nm.pred.value_counts().to_dict()
        res["per_label"][str(lab)] = d
    res["per_protocol"] = {}
    for proto, g in df.groupby(df.dport.map(protocol_of)):
        b, a = g[~g.is_attack], g[g.is_attack]
        res["per_protocol"][proto] = {"benign": rate(b.alert.sum(), len(b)), "attack": rate(a.alert.sum(), len(a))}
    cm = pd.crosstab(atk[atk.alert].true_class.fillna("(no SAINT class)"), atk[atk.alert].pred)
    res["confusion_detected_attacks"] = {str(i): {str(c): int(v) for c, v in row.items()} for i, row in cm.iterrows()}
    return res
