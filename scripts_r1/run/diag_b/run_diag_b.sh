#!/usr/bin/env bash
# SAINT-IDS Comment 13 — diagnostic B: the constant +40 ms per request with 2+ uvicorn workers.
# Reuses /path/to/workdir/venv, the X2 package (/path/to/workdir/transfer_x2: saint/ clone + x2_run.py, x2_receiver.py)
# and the latency sample. About 10 minutes.
# Usage:  cd /path/to/workdir/transfer_diag_b && mkdir -p logs && nohup bash run_diag_b.sh > logs/nohup.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=$PREV/venv/bin/python
X2=${X2:-$PREV/transfer_x2}
REPO=${REPO:-$X2/saint}
SAMPLE=${SAMPLE:-$PREV/transfer_latency/results/run1/latency_sample_v2.npz}
LOGS=$HERE/logs; OUT=$HERE/results; mkdir -p "$LOGS" "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps.log"; }
if pgrep -f "diag_b.py|x2_run.py" >/dev/null; then echo "a diagnostic/X2 run is already active - not starting another"; exit 1; fi
[ -f "$X2/x2_run.py" ] && [ -d "$REPO/.git" ] || { echo "X2 package not found at $X2 (set X2=...)"; exit 1; }
[ -f "$SAMPLE" ] || { echo "latency sample not found: $SAMPLE"; exit 1; }
log "commit $(git -C "$REPO" rev-parse --short HEAD); governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
{ uname -a; lscpu -e; uptime; sysctl net.ipv4.tcp_autocorking net.ipv4.tcp_low_latency 2>/dev/null; } > "$LOGS/machine_info.txt" 2>&1
export PYTHONDONTWRITEBYTECODE=1
unset ALERT_WEBHOOK_URL SAINT_MODELS_DIR SAINT_ALERT_POLICY CKKS_PUBLIC_CONTEXT CKKS_HMAC_SECRET
log "diagnostic B: cases C1-C7 (about 10 min)"
"$VPY" -u diag_b.py --repo "$REPO" --x2dir "$X2" --sample "$SAMPLE" --out "$OUT" \
    2>&1 | grep --line-buffered -v -i "VerifyOutputSizes\|UserWarning\|warnings.warn" | tee -a "$LOGS/diag_b.log"
tar -czf results_diag_b.tar.gz --exclude='results/keys' results logs
log "DONE -> $HERE/results_diag_b.tar.gz  (copy this file back to the laptop)"
