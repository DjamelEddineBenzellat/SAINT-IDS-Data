#!/usr/bin/env bash
# SAINT-IDS v2 (Comment 8) full training on a workstation — Linux / macOS / WSL.
# Usage:  bash run_v2.sh            (safe to re-run: every step resumes or skips)
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
BRANCH=004-cloud-ids-aggregation-engine
EXPECTED=$(cat EXPECTED_COMMIT)
REPO=$HERE/saint
CACHE=$HERE/cache            # prepared split + Optuna SQLite study (resume point)
LOGS=$HERE/logs
DATA=$HERE/data/cicids2017
PY=${PYTHON:-python3.12}
mkdir -p "$CACHE" "$LOGS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/run_v2_steps.log"; }

log "1/8 Python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || { echo "Need Python 3.12 (set PYTHON=/path/to/python3.12)"; exit 1; }

log "2/8 Repository from bundle"
if [ ! -d "$REPO/.git" ]; then
  git -c core.autocrlf=false clone -q -b "$BRANCH" saint_repo.bundle "$REPO"
fi
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }

log "3/8 Virtual environment (pinned versions)"
if [ ! -x "$HERE/venv/bin/python" ]; then
  "$PY" -m venv "$HERE/venv"
  "$HERE/venv/bin/pip" install -q --upgrade pip
  "$HERE/venv/bin/pip" install -q -r requirements_lock.txt
  "$HERE/venv/bin/pip" install -q tenseal==0.3.18 || log "   tenseal not installed (not needed for training)"
fi
VPY=$HERE/venv/bin/python
"$VPY" -m pip freeze > "$LOGS/pip_freeze_workstation.txt"

log "4/8 Dataset integrity (sha256 vs manifest)"
"$VPY" verify_v2.py --dataset "$DATA" | tee "$LOGS/dataset_check.txt"

log "5/8 Machine info"
{ uname -a; nproc; (lscpu 2>/dev/null | head -20) || true; free -g 2>/dev/null || true
  nvidia-smi 2>/dev/null || echo "no nvidia-smi"; } > "$LOGS/machine_info.txt" 2>&1

export SAINT_CICIDS_DIR=$DATA PYTHONDONTWRITEBYTECODE=1
cd "$REPO"

log "6/8 Unit tests for the v2 pipeline"
"$VPY" -m pytest tests/unit/test_train_pipeline_v2.py -q -p no:cacheprovider -o addopts="" 2>&1 | tee "$LOGS/pytest_v2.txt"

if [ ! -f edge/models_v2/training_report.json ] || ! grep -q onnx_fidelity_real_test edge/models_v2/training_report.json; then
  log "7/8 Prepare (dedup + split, cached) then full run: 50 trials, CPU, all cores"
  "$VPY" -m edge.training.train_pipeline_v2 --data-dir "$DATA" --cache-dir "$CACHE" --stage prepare \
      2>&1 | tee -a "$LOGS/v2_prepare.log"
  "$VPY" -m edge.training.train_pipeline_v2 --data-dir "$DATA" --cache-dir "$CACHE" --device cpu --n-jobs -1 \
      2>&1 | tee -a "$LOGS/v2_full.log"
else
  log "7/8 Training already finished — skipping"
fi

log "8/8 Verify + pack results"
cd "$HERE"
"$VPY" verify_v2.py "$REPO" | tee "$LOGS/verify_v2.txt"
tar -czf results_v2.tar.gz -C "$REPO" edge/models_v2 -C "$HERE" logs
log "DONE -> $HERE/results_v2.tar.gz  (copy this file back to the laptop)"
