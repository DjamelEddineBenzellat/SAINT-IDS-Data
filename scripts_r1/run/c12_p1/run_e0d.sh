#!/usr/bin/env bash
# SAINT-IDS Comment 12, E0d (~4 min): confirm the final extractor configuration
# (fin_existing, flags cicv3b = ACK of the first packet and FIN = SYN = RST = 0, frame payload, sample IAT std).
# Reuses the fin_existing extraction cached by run_e0c.sh (no PCAP pass).
# Usage:  cd /path/to/workdir/transfer_c12_p1 && nohup bash run_e0d.sh > logs/nohup_e0d.txt 2>&1 &
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
LOGS=$HERE/logs; OUT=$HERE/results_e0d; mkdir -p "$LOGS" "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps_e0d.log"; }
if pgrep -f "c12_e0.py|c12_e0b.py|c12_e1.py|c12_inventory.py" >/dev/null; then echo "a C12 run is already active - not starting another"; exit 1; fi
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }
[ -f "$WORK/e0c/e0c_flows_fin_existing.pkl" ] || echo "note: no cached extraction - E0d will extract once (~1.5 min more)"
export PYTHONDONTWRITEBYTECODE=1
log "1/2 E0 on fin_existing with the final flag rule (cached extraction)"
"$VPY" -u c12_e0.py --repo "$REPO" --pcap "$CICIDS_PCAP" --labelled "$CICIDS_LABELLED" --work "$WORK/e0c" --out "$OUT" \
    --fin-modes fin_existing 2>&1 | { grep --line-buffered -v "Column '.*' not found" || true; } | tee -a "$LOGS/e0d.log"
log "2/2 Pack"
tar -czf results_c12_e0d.tar.gz results_e0d logs/e0d.log logs/steps_e0d.log
log "DONE -> $HERE/results_c12_e0d.tar.gz  (copy this file back to the laptop)"
