#!/usr/bin/env bash
# SAINT-IDS Comment 13 — X0 (TenSEAL install) + X1 (real CKKS microbenchmarks) on the workstation.
# Reuses /path/to/workdir/venv, the repo clone in /path/to/workdir/transfer_e4/saint and the latency sample.
# Usage:  cd /path/to/workdir/transfer_x1 && mkdir -p logs && nohup bash run_x1.sh > logs/nohup.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=$PREV/venv/bin/python
REPO=${REPO:-$PREV/transfer_e4/saint}
SAMPLE=${SAMPLE:-$PREV/transfer_latency/results/run1/latency_sample_v2.npz}
CORE=${CORE:-2}
LOGS=$HERE/logs; OUT=$HERE/results; mkdir -p "$LOGS" "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps.log"; }
if pgrep -f c10_ckks_bench.py >/dev/null; then echo "an X1 run is already active - not starting another"; exit 1; fi

log "1/4 Inputs"
[ -x "$VPY" ] || { echo "venv not found at $PREV/venv"; exit 1; }
[ -d "$REPO/.git" ] || { echo "repo not found at $REPO (set REPO=...)"; exit 1; }
[ -f "$SAMPLE" ] || { echo "latency sample not found: $SAMPLE"; exit 1; }
git -C "$REPO" log --oneline -1 | tee "$LOGS/repo_commit.txt"

log "2/4 X0: TenSEAL 0.3.18"
if ! "$VPY" -c "import tenseal" 2>/dev/null; then
  "$VPY" -m pip install tenseal==0.3.18 > "$LOGS/pip_tenseal_online.txt" 2>&1 || {
    log "   online install failed (see logs/pip_tenseal_online.txt); trying the local wheel"
    WHL=$(ls "$HERE"/tenseal-0.3.18-*.whl 2>/dev/null | head -1 || true)
    [ -n "$WHL" ] || { echo "no local wheel: copy tenseal-0.3.18-*.whl into $HERE"; exit 1; }
    "$VPY" -m pip install --no-deps "$WHL" > "$LOGS/pip_tenseal_wheel.txt" 2>&1
  }
fi
"$VPY" -c "import tenseal, sys; assert 'tenseal_sim' not in tenseal.__file__; print('TenSEAL', tenseal.__version__, tenseal.__file__, sys.version)" | tee "$LOGS/tenseal_check.txt"
{ uname -a; lscpu | head -20; uptime; } > "$LOGS/machine_info.txt" 2>&1

log "3/4 X1: CKKS microbenchmarks (1 core; about 3-5 min)"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$VPY" -u c10_ckks_bench.py --repo "$REPO" --sample "$SAMPLE" --out "$OUT" --core "$CORE" \
    2>&1 | grep --line-buffered -v -i "VerifyOutputSizes" | tee -a "$LOGS/x1.log"
log "4/4 Pack results"
tar -czf results_x1.tar.gz results logs
log "DONE -> $HERE/results_x1.tar.gz  (copy this file back to the laptop)"
