#!/usr/bin/env bash
# SAINT-IDS Comment 12: CICIoMT2024 inventory BEFORE any scoring (the label map is built from it).
# Parses only what E1 needs: WiFI_and_MQTT/attacks/PCAP/test/* and WiFI_and_MQTT/profiling/PCAP/*.
# Everything else (train, Bluetooth, CSV, the .tar.xz) is only listed with its size.
# Usage:  cd /path/to/workdir/transfer_c12_p1 && nohup bash run_inventory.sh > logs/nohup_inv.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=${VPY:-$PREV/venv/bin/python}
CICIOMT_ROOT=${CICIOMT_ROOT:-$PREV/data/CICIoMT2024}
JOBS=${JOBS:-4}
LOGS=$HERE/logs; OUT=$HERE/results_inv; mkdir -p "$LOGS" "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps_inv.log"; }
if pgrep -f "c12_e0.py|c12_e0b.py|c12_e1.py|c12_inventory.py" >/dev/null; then echo "a C12 run is already active - not starting another"; exit 1; fi
[ -d "$CICIOMT_ROOT/WiFI_and_MQTT" ] || { echo "no WiFI_and_MQTT folder under $CICIOMT_ROOT (set CICIOMT_ROOT=...)"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
log "1/2 Inventory of $CICIOMT_ROOT (test + profiling captures: SHA-256, packets, top talkers; $JOBS jobs)"
{ echo "--- folder sizes"; du -sh "$CICIOMT_ROOT"/* "$CICIOMT_ROOT"/WiFI_and_MQTT/* "$CICIOMT_ROOT"/WiFI_and_MQTT/*/* 2>/dev/null || true; } | tee -a "$LOGS/steps_inv.log"
"$VPY" -u c12_inventory.py --root "$CICIOMT_ROOT" --out "$OUT" --jobs "$JOBS" \
    --include "WiFI_and_MQTT/attacks/PCAP/test/*" --include "WiFI_and_MQTT/profiling/PCAP/*" 2>&1 | tee -a "$LOGS/inventory.log"
log "2/2 Pack"
tar -czf results_c12_inv.tar.gz results_inv logs/inventory.log logs/steps_inv.log
log "DONE -> $HERE/results_c12_inv.tar.gz  (copy this file back to the laptop)"
