#!/usr/bin/env bash
# SAINT-IDS latency benchmarks (V1 vs v2 models) on the workstation.
# Reuses the venv and the prepared-data cache from the training package (/path/to/workdir).
# Usage:  bash run_latency.sh            Override: PREV=/path/to/transfer CORE=2 RUNS=3 bash run_latency.sh
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}            # training package folder (venv/ + cache/)
VPY=$PREV/venv/bin/python
CACHE=$PREV/cache
CORE=${CORE:-2}                         # core for the single-thread blocks B and C
RUNS=${RUNS:-3}
BRANCH=004-cloud-ids-aggregation-engine
EXPECTED=$(cat EXPECTED_COMMIT)
REPO=$HERE/saint
LOGS=$HERE/logs; mkdir -p "$LOGS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps.log"; }

log "1/6 Reuse venv + cache from $PREV"
[ -x "$VPY" ] || { echo "venv not found at $PREV/venv (set PREV=...)"; exit 1; }
ls "$CACHE"/prepared_*.npz >/dev/null || { echo "no prepared_*.npz in $CACHE"; exit 1; }

log "2/6 Repository from bundle (commit ${EXPECTED:0:7})"
[ -d "$REPO/.git" ] || git -c core.autocrlf=false clone -q -b "$BRANCH" saint_repo.bundle "$REPO"
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }

log "3/6 Policy/model binding + decision tests"
(cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 "$VPY" -m pytest tests/unit/test_decision.py -q -p no:cacheprovider -o addopts="" 2>&1 | tail -1) | tee "$LOGS/pytest_decision.txt"

log "4/6 Machine state"
{ uname -a; lscpu | head -20; echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
  echo "no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"; uptime; } > "$LOGS/machine_info.txt" 2>&1
echo "   Close other heavy programs now; timings are single-core on CPU $CORE."

export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset ALERT_WEBHOOK_URL SAINT_MODELS_DIR SAINT_ALERT_POLICY
for r in $(seq 1 "$RUNS"); do
  log "5/6 Benchmark run $r/$RUNS (about 3-5 min)"
  if [ -n "$(ls "$HERE/results/run$r"/latency_*.json 2>/dev/null)" ]; then log "   run $r already done - skipping"; continue; fi
  [ -f "$HERE/results/run1/latency_sample_v2.npz" ] && mkdir -p "$HERE/results/run$r" && cp -n "$HERE/results/run1/latency_sample_v2.npz" "$HERE/results/run$r/" 2>/dev/null || true
  # CPU pinning is done inside the script (block A unpinned, B/C on one core); -u = unbuffered output
  "$VPY" -u c7_latency_v2.py --repo "$REPO" --cache "$CACHE" --out "$HERE/results/run$r" --threads 1 --core "$CORE" \
      2>&1 | grep --line-buffered -v -i "VerifyOutputSizes" | tee "$LOGS/run$r.log"
done

log "6/6 Pack results"
tar -czf results_latency.tar.gz results logs
log "DONE -> $HERE/results_latency.tar.gz  (copy this file back to the laptop)"
