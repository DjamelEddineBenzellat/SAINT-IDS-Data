"""Replay the 16 demo/iot_evasion_suite.ps1 scenarios through the C′ API path (temp logs)."""
import re, sys, tempfile, json
from pathlib import Path
ROOT = Path("/path/to/SAINT/realtime-ids-healthcareBeta"); sys.path.insert(0, str(ROOT))
from edge.api.handlers import Handlers
from edge.api.schemas import PacketScan
from edge.inference.alert_router import AlertRouter
from edge.inference.flow_ingestor import FlowIngestor
from edge.inference.scoring_engine import ScoringEngine
src = (ROOT / "demo/iot_evasion_suite.ps1").read_text(encoding="utf-8", errors="replace")
h = Handlers(flow_ingestor=FlowIngestor(), scoring_engine=ScoringEngine(models_dir=ROOT / "edge/models"),
             alert_router=AlertRouter(log_dir=tempfile.mkdtemp(), max_alerts_per_minute=10**6))
rows = []
for blk in re.split(r"\n(?=Send-HealthcareTest )", src)[1:]:
    blk = blk.split("\n# ===")[0]
    g = lambda k: re.search(rf'-{k}\s+"([^"]*)"', blk).group(1)
    port = int(re.search(r"-Port\s+(\d+)", blk).group(1))
    meta = {"Destination Port": port, "Dst Port": port, "Protocol": int(re.search(r"-Protocol\s+(\d+)", blk).group(1)), "Src Port": 40000}
    meta.update({m.group(1): float(m.group(2)) for m in re.finditer(r'"([A-Za-z_]+)"\s*=\s*(-?[0-9.]+)', blk)})
    r = h._scan_single(PacketScan(raw_packet=g("RawHex"), source_ip=g("IP"), metadata=meta))
    rows.append({"scenario": g("Name"), "category": g("Category"), "expected": g("ExpectedClass"),
                 "should_detect": "$true" in blk.split("-ShouldDetect")[1][:8], "severity": r.severity,
                 "predicted": r.predicted_class, "s_attack": round(r.threat_score, 6), "sources": r.alert_sources,
                 "encrypt": r.encrypt, "policy": r.policy_version})
out = AUD / "reports/policy_decision/healthcare_replay_cprime.json" if (AUD := Path("/path/to/SAINT/revision-audit")) else None
out.write_text(json.dumps(rows, indent=1))
atk = [r for r in rows if r["should_detect"]]; ben = [r for r in rows if not r["should_detect"]]
print(f"attacks detected {sum(r['severity']!='INFO' for r in atk)}/{len(atk)}; class correct {sum(r['predicted']==r['expected'] for r in atk)}/{len(atk)}; baseline false alarms {sum(r['severity']!='INFO' for r in ben)}/{len(ben)}")
for r in rows: print(f"{r['category']:11s} {r['scenario'][:36]:36s} exp={r['expected']:17s} got={r['predicted']:17s} {r['severity']:8s} s={r['s_attack']:<8} {r['sources']}")
