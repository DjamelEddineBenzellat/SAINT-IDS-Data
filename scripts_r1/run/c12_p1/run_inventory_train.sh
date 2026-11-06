#!/usr/bin/env bash
# SAINT-IDS Comment 12: CICIoMT2024 inventory BEFORE any scoring (the label map is built from it).
# E1b P1: inventory of the 51 Wi-Fi/MQTT TRAIN captures (attacking hosts, SHA-256, packets) before any labelling or training.
# Everything else (test, profiling, Bluetooth, CSV, the .tar.xz) is only listed with its size. Nothing is scored or trained.
# Usage:  cd /path/to/workdir/transfer_c12_p1 && nohup bash run_inventory_train.sh > logs/nohup_inv_train.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=${VPY:-$PREV/venv/bin/python}
CICIOMT_ROOT=${CICIOMT_ROOT:-$PREV/data/CICIoMT2024}
JOBS=${JOBS:-4}
LOGS=$HERE/logs; OUT=$HERE/results_inv_train; mkdir -p "$LOGS" "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps_inv_train.log"; }
if pgrep -f "c12_e0.py|c12_e0b.py|c12_e1.py|c12_inventory.py" >/dev/null; then echo "a C12 run is already active - not starting another"; exit 1; fi
[ -d "$CICIOMT_ROOT/WiFI_and_MQTT" ] || { echo "no WiFI_and_MQTT folder under $CICIOMT_ROOT (set CICIOMT_ROOT=...)"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
log "1/2 Inventory of the train captures in $CICIOMT_ROOT (56.7 GB; SHA-256, packets, top talkers; $JOBS jobs; ~10 min)"
{ echo "--- folder sizes"; du -sh "$CICIOMT_ROOT"/* "$CICIOMT_ROOT"/WiFI_and_MQTT/* "$CICIOMT_ROOT"/WiFI_and_MQTT/*/* 2>/dev/null || true; } | tee -a "$LOGS/steps_inv_train.log"
"$VPY" -u c12_inventory.py --root "$CICIOMT_ROOT" --out "$OUT" --jobs "$JOBS" \
    --include "WiFI_and_MQTT/attacks/PCAP/train/*" 2>&1 | tee -a "$LOGS/inventory_train.log"
log "2/2 Pack"
tar -czf results_c12_inv_train.tar.gz results_inv_train logs/inventory_train.log logs/steps_inv_train.log
log "DONE -> $HERE/results_c12_inv_train.tar.gz  (copy this file back to the laptop)"
