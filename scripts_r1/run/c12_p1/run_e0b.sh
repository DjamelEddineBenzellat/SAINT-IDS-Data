#!/usr/bin/env bash
# SAINT-IDS Comment 12, E0b diagnostic (~5 min): which feature DEFINITIONS reproduce the published CICIDS-2017
# Friday CSV? One extraction pass with directional / per-packet variants, then definition tests on matched pairs.
# Usage:  cd /path/to/workdir/transfer_c12_p1 && nohup bash run_e0b.sh > logs/nohup_e0b.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=${VPY:-$PREV/venv/bin/python}
REPO=${REPO:-$PREV/transfer_c9/saint}
DATA_C12=${DATA_C12:-$PREV/data_c12}
CICIDS_PCAP=${CICIDS_PCAP:-$DATA_C12/cicids2017_pcap/Friday-WorkingHours.pcap}
CICIDS_LABELLED=${CICIDS_LABELLED:-$DATA_C12/cicids2017_labelled}
WORK=${WORK:-$PREV/work_c12}
EXPECTED=$(cat EXPECTED_COMMIT)
LOGS=$HERE/logs; OUT=$HERE/results_e0b; mkdir -p "$LOGS" "$OUT" "$WORK/e0b"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps_e0b.log"; }
if pgrep -f "c12_e0.py|c12_e0b.py|c12_e1.py|c12_inventory.py" >/dev/null; then echo "a C12 run is already active - not starting another"; exit 1; fi
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
log "1/3 Self-test"
"$VPY" -u c12_selftest.py --repo "$REPO" --tmp "$WORK/selftest" 2>&1 | tee -a "$LOGS/selftest.log"
log "2/3 E0b: extraction with definition variants + tests + re-score (~5 min)"
"$VPY" -u c12_e0b.py --repo "$REPO" --pcap "$CICIDS_PCAP" --labelled "$CICIDS_LABELLED" --work "$WORK/e0b" --out "$OUT" \
    2>&1 | { grep --line-buffered -v "Column '.*' not found" || true; } | tee -a "$LOGS/e0b.log"
log "3/3 Pack"
tar -czf results_c12_e0b.tar.gz results_e0b logs/e0b.log logs/steps_e0b.log logs/selftest.log
log "DONE -> $HERE/results_c12_e0b.tar.gz  (copy this file back to the laptop)"
