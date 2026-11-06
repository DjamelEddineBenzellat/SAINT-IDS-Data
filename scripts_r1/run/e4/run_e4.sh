#!/usr/bin/env bash
# SAINT-IDS E4 (Comment 4 step 1 verification) on the workstation:
#   full unit + integration tests, dataset regression (exact counts), latency of the new path x3.
# Reuses /path/to/workdir/venv, /path/to/workdir/data/cicids2017 and the latency sample. Safe to re-run.
# Usage:  cd /path/to/workdir/transfer_e4 && mkdir -p logs && nohup bash run_e4.sh > logs/nohup.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=$PREV/venv/bin/python
DATA=${DATA:-$PREV/data/cicids2017}
SAMPLE=${SAMPLE:-$PREV/transfer_latency/results/run1/latency_sample_v2.npz}
CORE=${CORE:-2}; RUNS=${RUNS:-3}
BRANCH=004-cloud-ids-aggregation-engine
EXPECTED=$(cat EXPECTED_COMMIT)
REPO=$HERE/saint
LOGS=$HERE/logs; OUT=$HERE/results; mkdir -p "$LOGS" "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps.log"; }
if pgrep -f c9_latency_e4.py >/dev/null; then echo "an E4 run is already active - not starting another"; exit 1; fi

log "1/6 Inputs"
[ -x "$VPY" ] || { echo "venv not found at $PREV/venv"; exit 1; }
[ -f "$SAMPLE" ] || { echo "latency sample not found: $SAMPLE"; exit 1; }
ls "$DATA"/*.csv >/dev/null || { echo "dataset not found: $DATA"; exit 1; }
log "2/6 Repository (commit ${EXPECTED:0:7})"
# Small update bundle (only the commits after d83a6f1) applied on top of the clone already on this
# machine (/path/to/workdir/transfer_latency/saint or transfer_step1/saint); no large file to copy.
BASE=${BASE:-}
for b in "$PREV/transfer_step1/saint" "$PREV/transfer_latency/saint"; do [ -z "$BASE" ] && [ -d "$b/.git" ] && BASE=$b; done
if [ ! -d "$REPO/.git" ]; then
  sha256sum -c saint_update.bundle.sha256 || { echo "update bundle is incomplete/corrupt - copy it again"; exit 1; }
  [ -n "$BASE" ] || { echo "no existing clone found under $PREV (set BASE=/path/to/saint)"; exit 1; }
  git -c core.autocrlf=false clone -q "$BASE" "$REPO"
  git -C "$REPO" fetch -q "$HERE/saint_update.bundle" "$BRANCH"
  git -C "$REPO" checkout -q -B "$BRANCH" FETCH_HEAD
fi
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }
{ uname -a; lscpu | head -20; uptime; } > "$LOGS/machine_info.txt" 2>&1
export PYTHONDONTWRITEBYTECODE=1
unset ALERT_WEBHOOK_URL SAINT_MODELS_DIR SAINT_ALERT_POLICY DOCKER_STACK_URL

log "3/6 Unit + integration tests"
(cd "$REPO" && "$VPY" -m pytest tests/unit tests/integration -q -p no:cacheprovider -o addopts="" --timeout=300 2>&1 | tail -3) | tee "$LOGS/pytest_unit_integration.txt" || true
log "4/6 Dataset regression (deployed policy 2.1.0 on the v2 test split + historical V1 lock)"
(cd "$REPO" && SAINT_CICIDS_DIR="$DATA" "$VPY" -m pytest tests/regression -v -p no:cacheprovider -o addopts="" 2>&1 | tail -6) | tee "$LOGS/pytest_regression.txt" || true
git -C "$REPO" status --short > "$LOGS/repo_status_after_tests.txt"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for r in $(seq 1 "$RUNS"); do
  log "5/6 Latency run $r/$RUNS (about 1-2 min)"
  "$VPY" -u c9_latency_e4.py --repo "$REPO" --sample "$SAMPLE" --out "$OUT/run$r" --core "$CORE" \
      2>&1 | grep --line-buffered -v -i "VerifyOutputSizes" | tee -a "$LOGS/run$r.log"
done
log "6/6 Pack results"
tar -czf results_e4.tar.gz results logs
log "DONE -> $HERE/results_e4.tar.gz  (copy this file back to the laptop)"
