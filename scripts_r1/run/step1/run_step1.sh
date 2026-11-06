#!/usr/bin/env bash
# SAINT-IDS step 1 evidence (Comment 4 + SHAP): E1 IF ablation, E2 unseen-class test, E3 SHAP methods.
# Reuses /path/to/workdir/venv and /path/to/workdir/cache. Safe to re-run: finished stages are skipped.
# Usage:  nohup bash run_step1.sh > logs/nohup.txt 2>&1 &      then   tail -f logs/nohup.txt
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=$PREV/venv/bin/python
CACHE=$PREV/cache
CORE=${CORE:-2}
BRANCH=004-cloud-ids-aggregation-engine
EXPECTED=$(cat EXPECTED_COMMIT)
REPO=$HERE/saint
LOGS=$HERE/logs; OUT=$HERE/results; mkdir -p "$LOGS" "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps.log"; }

if pgrep -f c8_step1_evidence.py >/dev/null; then echo "a step-1 run is already active - not starting another"; exit 1; fi
log "1/5 Reuse venv + cache from $PREV"
[ -x "$VPY" ] || { echo "venv not found at $PREV/venv (set PREV=...)"; exit 1; }
ls "$CACHE"/prepared_*.npz >/dev/null || { echo "no prepared_*.npz in $CACHE"; exit 1; }
log "2/5 Repository (commit ${EXPECTED:0:7})"
[ -d "$REPO/.git" ] || git -c core.autocrlf=false clone -q -b "$BRANCH" saint_repo.bundle "$REPO"
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }
{ uname -a; lscpu | head -20; uptime; } > "$LOGS/machine_info.txt" 2>&1
export PYTHONDONTWRITEBYTECODE=1
unset ALERT_WEBHOOK_URL SAINT_MODELS_DIR SAINT_ALERT_POLICY
run() {  # $1 stage, $2 label
  log "$2"
  "$VPY" -u c8_step1_evidence.py --repo "$REPO" --cache "$CACHE" --out "$OUT" --stage "$1" --core "$CORE" \
      2>&1 | grep --line-buffered -v -i "VerifyOutputSizes" | tee -a "$LOGS/$1.log"
}
run e1 "3/5 E1 Isolation Forest ablation (all cores, ~2-4 min)"
run e2 "4/5 E2 unseen-attack-class test: 5 refits (all cores, ~5-10 min)"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 run e3 "5/5 E3 SHAP methods on 1,000 flows (1 core; M0 is slow, ~6-8 min)"
tar -czf results_step1.tar.gz results logs
log "DONE -> $HERE/results_step1.tar.gz  (copy this file back to the laptop)"
