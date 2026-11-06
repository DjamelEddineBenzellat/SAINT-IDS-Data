"""
C6 — Replay the 16 demo/iot_evasion_suite.ps1 scenarios through the deployed API path
(Handlers → ScoringEngine → decide → AlertRouter) with the CURRENT policy file
(2.0.0, τ = 0.25 FINAL) and its bound serving model (edge/models_v2). Temp logs only.
Compares with the V1-model replay under policy 1.0.0 (reports/policy_decision/healthcare_replay_cprime.json).
Same scenario parsing as c3_healthcare_replay_cprime.py.
"""
import json
import re
import sys
import tempfile
from pathlib import Path

AUD = Path(__file__).resolve().parents[1]
ROOT = AUD.parent / "realtime-ids-healthcareBeta"
sys.path.insert(0, str(ROOT))
from edge.api.handlers import Handlers  # noqa: E402
from edge.api.schemas import PacketScan  # noqa: E402
from edge.inference.alert_router import AlertRouter  # noqa: E402
from edge.inference.decision import get_default_policy, serving_models_dir  # noqa: E402
from edge.inference.flow_ingestor import FlowIngestor  # noqa: E402
from edge.inference.scoring_engine import ScoringEngine  # noqa: E402

pol = get_default_policy()
se = ScoringEngine(models_dir=serving_models_dir(pol), policy=pol)
binding = se.get_model_info()["model_binding"]
assert binding["match"] is True, binding
h = Handlers(flow_ingestor=FlowIngestor(), scoring_engine=se,
             alert_router=AlertRouter(log_dir=tempfile.mkdtemp(), max_alerts_per_minute=10**6))

src = (ROOT / "demo/iot_evasion_suite.ps1").read_text(encoding="utf-8", errors="replace")
rows = []
for blk in re.split(r"\n(?=Send-HealthcareTest )", src)[1:]:
    blk = blk.split("\n# ===")[0]
    g = lambda k: re.search(rf'-{k}\s+"([^"]*)"', blk).group(1)  # noqa: E731
    port = int(re.search(r"-Port\s+(\d+)", blk).group(1))
    meta = {"Destination Port": port, "Dst Port": port, "Protocol": int(re.search(r"-Protocol\s+(\d+)", blk).group(1)),
            "Src Port": 40000}
    meta.update({m.group(1): float(m.group(2)) for m in re.finditer(r'"([A-Za-z_]+)"\s*=\s*(-?[0-9.]+)', blk)})
    r = h._scan_single(PacketScan(raw_packet=g("RawHex"), source_ip=g("IP"), metadata=meta))
    rows.append({"scenario": g("Name"), "category": g("Category"), "expected": g("ExpectedClass"),
                 "should_detect": "$true" in blk.split("-ShouldDetect")[1][:8], "severity": r.severity,
                 "predicted": r.predicted_class, "s_attack": r.threat_score, "sources": r.alert_sources,
                 "ml_alert": r.threat_score >= pol.alert_threshold, "encrypt": r.encrypt, "policy": r.policy_version})

v1 = {r["scenario"]: r for r in json.loads((AUD / "reports/policy_decision/healthcare_replay_cprime.json").read_text())}
for r in rows:
    o = v1.get(r["scenario"], {})
    r["v1_policy1_severity"] = o.get("severity"); r["v1_s_attack"] = o.get("s_attack"); r["v1_sources"] = o.get("sources")

atk = [r for r in rows if r["should_detect"]]; ben = [r for r in rows if not r["should_detect"]]
summary = {
    "policy": pol.describe(), "model_binding": binding,
    "attacks": len(atk), "baselines": len(ben),
    "detected": sum(r["severity"] != "INFO" for r in atk),
    "detected_by_ml": sum(r["ml_alert"] for r in atk),
    "detected_by_rules_only": sum(r["severity"] != "INFO" and not r["ml_alert"] for r in atk),
    "class_correct": sum(r["predicted"] == r["expected"] for r in atk),
    "baseline_false_alarms": sum(r["severity"] != "INFO" for r in ben),
    "max_s_attack_on_attacks": max(r["s_attack"] for r in atk),
    "v1_policy1_detected": sum(r["v1_policy1_severity"] not in (None, "INFO") for r in atk),
}
out = AUD / "reports/policy_decision/healthcare_replay_v2_tau025.json"
out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str))
print(json.dumps({k: v for k, v in summary.items() if k not in ("policy", "model_binding")}))
for r in rows:
    print(f"{r['category']:11s} {r['scenario'][:36]:36s} exp={r['expected']:17s} got={r['predicted']:17s} "
          f"{r['severity']:8s} s={r['s_attack']:.3g} (V1 {r['v1_s_attack']}) {r['sources']}  V1:{r['v1_policy1_severity']}")
