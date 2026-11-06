#!/usr/bin/env bash
# SAINT-IDS Comment 12, E0c (~10 min): E0 again with the two rules found by E0b
#   - payload = captured frame bytes after the L4 header (Ethernet padding counted, as CICFlowMeter-V3 did)
#   - a FIN that STARTS a flow does not close it (fin_mode fin_existing)
#   - flags "cicv3": FIN / ACK of the first packet, SYN = RST = 0
# then the E0b diagnostic on fin_existing, to see what still disagrees.
# Usage:  cd /path/to/workdir/transfer_c12_p1 && nohup bash run_e0c.sh > logs/nohup_e0c.txt 2>&1 &
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
LOGS=$HERE/logs; OUT=$HERE/results_e0c; mkdir -p "$LOGS" "$OUT" "$WORK/e0c"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps_e0c.log"; }
if pgrep -f "c12_e0.py|c12_e0b.py|c12_e1.py|c12_inventory.py" >/dev/null; then echo "a C12 run is already active - not starting another"; exit 1; fi
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
log "1/4 Self-test"
"$VPY" -u c12_selftest.py --repo "$REPO" --tmp "$WORK/selftest" 2>&1 | tee -a "$LOGS/selftest.log"
log "2/4 E0 with the new rules: 2 extraction passes, 12 variants each, model re-score (~6 min)"
"$VPY" -u c12_e0.py --repo "$REPO" --pcap "$CICIDS_PCAP" --labelled "$CICIDS_LABELLED" --work "$WORK/e0c" --out "$OUT" \
    2>&1 | { grep --line-buffered -v "Column '.*' not found" || true; } | tee -a "$LOGS/e0c.log"
log "3/4 E0b diagnostic on fin_existing (~4 min)"
"$VPY" -u c12_e0b.py --repo "$REPO" --pcap "$CICIDS_PCAP" --labelled "$CICIDS_LABELLED" --work "$WORK/e0c" --out "$OUT" \
    --fin-mode fin_existing 2>&1 | { grep --line-buffered -v "Column '.*' not found" || true; } | tee -a "$LOGS/e0c_diag.log"
log "4/4 Pack"
tar -czf results_c12_e0c.tar.gz results_e0c logs/e0c.log logs/e0c_diag.log logs/steps_e0c.log logs/selftest.log
log "DONE -> $HERE/results_c12_e0c.tar.gz  (copy this file back to the laptop)"
