#!/usr/bin/env bash
# SAINT-IDS Comment 3, G2 (~3-5 min): per raw CICIDS-2017 label recall of the deployed model on the v2 test split
# (content-type attacks: XSS, SQL injection, web brute force, Infiltration, Heartbleed, Bot). Read-only.
# Usage:  cd /path/to/workdir/transfer_c3 && mkdir -p logs && nohup bash run_c3.sh > logs/nohup.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=${VPY:-$PREV/venv/bin/python}
REPO=${REPO:-$PREV/transfer_c12_p2/saint}
DATA=${DATA:-$PREV/data/cicids2017}
CACHE=${CACHE:-$PREV/cache}
LOGS=$HERE/logs; OUT=$HERE/results; mkdir -p "$LOGS" "$OUT"
[ -d "$REPO/.git" ] || { echo "no repo at $REPO (set REPO=...)"; exit 1; }
ls "$CACHE"/prepared_*.npz >/dev/null || { echo "no prepared cache in $CACHE"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
unset SAINT_MODELS_DIR SAINT_ALERT_POLICY
echo "[$(date '+%F %T')] repo $(git -C "$REPO" rev-parse --short HEAD); per-raw-label recall" | tee "$LOGS/steps.log"
"$VPY" -u c3_rawlabel_recall.py --repo "$REPO" --data "$DATA" --cache "$CACHE" --out "$OUT" 2>&1 \
    | { grep --line-buffered -v "Column '.*' not found" || true; } | tee "$LOGS/c3.log"
tar -czf results_c3.tar.gz results logs
echo "[$(date '+%F %T')] DONE -> $HERE/results_c3.tar.gz" | tee -a "$LOGS/steps.log"
