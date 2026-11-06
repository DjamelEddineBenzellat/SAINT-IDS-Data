#!/usr/bin/env bash
# SAINT-IDS Comment 12, package P1 (part b): E1 on CICIoMT2024 (v2 model, tau = 0.25 unchanged).
# label_map.json (fixed from the inventory, 2026-10-30) and e0_extractor_config.json (frozen by E0d) are in this folder.
# Expect ~40-60 min: extraction of ~159 M packets (sequential) + scoring of the sampled flows + 1000-flow parity.
# Usage:  cd /path/to/workdir/transfer_c12_p1 && nohup bash run_p1b.sh > logs/nohup_b.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=${VPY:-$PREV/venv/bin/python}
REPO=${REPO:-$PREV/transfer_c9/saint}
CICIOMT_ROOT=${CICIOMT_ROOT:-$PREV/data/CICIoMT2024}
WORK=${WORK:-$PREV/work_c12}
JOBS=${JOBS:-1}            # 1 = sequential: flood captures can hold millions of active flows in RAM
MAX_FLOWS=${MAX_FLOWS:-200000}   # uniform reservoir sample per capture (weights recorded)
EXPECTED=$(cat EXPECTED_COMMIT)
LOGS=$HERE/logs; OUT=$HERE/results_b; mkdir -p "$LOGS" "$OUT" "$WORK/e1"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps_b.log"; }
if pgrep -f "c12_e0.py|c12_e1.py|c12_inventory.py" >/dev/null; then echo "a C12 run is already active - not starting another"; exit 1; fi
for f in label_map.json e0_extractor_config.json; do [ -f "$f" ] || { echo "missing $f (sent from the laptop)"; exit 1; }; done
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 IP_PSEUDONYM_KEY=c12-parity-only
unset ALERT_WEBHOOK_URL SAINT_MODELS_DIR SAINT_ALERT_POLICY CKKS_PUBLIC_CONTEXT CKKS_HMAC_SECRET DOCKER_STACK_URL
cp label_map.json e0_extractor_config.json "$OUT/"
{ echo "--- memory / disk at start"; free -g; df -h "$WORK"; } | tee -a "$LOGS/steps_b.log"

log "1/3 Extract flows from the CICIoMT2024 Wi-Fi/MQTT captures ($JOBS jobs; cached per capture, resumable)"
"$VPY" -u c12_e1.py extract --root "$CICIOMT_ROOT" --label-map label_map.json --extractor e0_extractor_config.json \
    --work "$WORK/e1" --jobs "$JOBS" --max-flows "$MAX_FLOWS" 2>&1 | tee -a "$LOGS/e1_extract.log"
log "2/3 Score every flow through the deployed decision path (+ 1000-flow full-handler parity)"
"$VPY" -u c12_e1.py score --repo "$REPO" --label-map label_map.json --extractor e0_extractor_config.json \
    --work "$WORK/e1" --out "$OUT" --parity 1000 2>&1 | { grep --line-buffered -v "Column '.*' not found" || true; } | tee -a "$LOGS/e1_score.log"
log "3/3 Pack results (the per-flow decision table stays here: $OUT/e1_decisions.csv.gz)"
tar -czf results_c12_p1b.tar.gz --exclude='results_b/e1_decisions.csv.gz' results_b logs
log "DONE -> $HERE/results_c12_p1b.tar.gz  (copy this file back to the laptop)"
