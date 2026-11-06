#!/usr/bin/env bash
# SAINT-IDS Comment 12, E1 run 2 (~5 min): re-score the CACHED CICIoMT2024 flows with the corrected labelling
# (attacker <-> victim in EITHER direction) and report the alerting views (full / ML only / ML + fast filter /
# evasion rules only) and the model's own naming. No PCAP pass. Same model, tau and policy file.
# Usage:  cd /path/to/workdir/transfer_c12_p1 && nohup bash run_e1_rescore.sh > logs/nohup_e1r2.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=${VPY:-$PREV/venv/bin/python}
REPO=${REPO:-$PREV/transfer_c9/saint}
WORK=${WORK:-$PREV/work_c12}
EXPECTED=$(cat EXPECTED_COMMIT)
LOGS=$HERE/logs; OUT=$HERE/results_b2; mkdir -p "$LOGS" "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps_b2.log"; }
if pgrep -f "c12_e0.py|c12_e1.py|c12_inventory.py" >/dev/null; then echo "a C12 run is already active - not starting another"; exit 1; fi
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 IP_PSEUDONYM_KEY=c12-parity-only
unset ALERT_WEBHOOK_URL SAINT_MODELS_DIR SAINT_ALERT_POLICY CKKS_PUBLIC_CONTEXT CKKS_HMAC_SECRET DOCKER_STACK_URL
cp label_map.json e0_extractor_config.json "$OUT/"
log "1/2 Score the cached flows (corrected labelling) + 1000-flow full-handler parity"
"$VPY" -u c12_e1.py score --repo "$REPO" --label-map label_map.json --extractor e0_extractor_config.json \
    --work "$WORK/e1" --out "$OUT" --parity 1000 2>&1 | { grep --line-buffered -v "Column '.*' not found" || true; } | tee -a "$LOGS/e1_score2.log"
log "2/2 Pack (per-flow table stays here)"
tar -czf results_c12_e1r2.tar.gz --exclude='results_b2/e1_decisions.csv.gz' results_b2 logs/e1_score2.log logs/steps_b2.log
log "DONE -> $HERE/results_c12_e1r2.tar.gz  (copy this file back to the laptop)"
