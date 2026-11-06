#!/usr/bin/env bash
# SAINT-IDS Comment 13 — X2 (HTTP load tests with CKKS) + X3 (component comparison) on the workstation.
# Reuses /path/to/workdir/venv (TenSEAL installed by X0), /path/to/workdir/cache, the latency sample and the
# clone in /path/to/workdir/transfer_e4/saint (updated here with a small bundle). Safe to re-run.
# Usage:  cd /path/to/workdir/transfer_x2 && mkdir -p logs && nohup bash run_x2.sh > logs/nohup.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=$PREV/venv/bin/python
CACHE=${CACHE:-$PREV/cache}
SAMPLE=${SAMPLE:-$PREV/transfer_latency/results/run1/latency_sample_v2.npz}
BASE=${BASE:-$PREV/transfer_e4/saint}
CORE=${CORE:-2}
BRANCH=004-cloud-ids-aggregation-engine
EXPECTED=$(cat EXPECTED_COMMIT)
REPO=$HERE/saint
LOGS=$HERE/logs; OUT=$HERE/results; mkdir -p "$LOGS" "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps.log"; }
if pgrep -f "x2_run.py|x3_offline.py" >/dev/null; then echo "an X2/X3 run is already active - not starting another"; exit 1; fi

log "1/6 Inputs"
[ -x "$VPY" ] || { echo "venv not found at $PREV/venv"; exit 1; }
"$VPY" -c "import tenseal, httpx, psutil, uvicorn; assert 'tenseal_sim' not in tenseal.__file__; print('tenseal', tenseal.__version__)" | tee "$LOGS/env_check.txt"
[ -f "$SAMPLE" ] || { echo "latency sample not found: $SAMPLE"; exit 1; }
ls "$CACHE"/prepared_*.npz >/dev/null || { echo "no prepared_*.npz in $CACHE"; exit 1; }
log "2/6 Repository (commit ${EXPECTED:0:7})"
if [ ! -d "$REPO/.git" ]; then
  sha256sum -c saint_update.bundle.sha256 || { echo "update bundle is incomplete/corrupt - copy it again"; exit 1; }
  [ -d "$BASE/.git" ] || { echo "no base clone at $BASE (set BASE=...)"; exit 1; }
  git -c core.autocrlf=false clone -q "$BASE" "$REPO"
  git -C "$REPO" fetch -q "$HERE/saint_update.bundle" "$BRANCH"
  git -C "$REPO" checkout -q -B "$BRANCH" FETCH_HEAD
fi
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }
{ uname -a; lscpu | head -24; lscpu -e; uptime; } > "$LOGS/machine_info.txt" 2>&1
export PYTHONDONTWRITEBYTECODE=1
unset ALERT_WEBHOOK_URL SAINT_MODELS_DIR SAINT_ALERT_POLICY CKKS_PUBLIC_CONTEXT CKKS_HMAC_SECRET DOCKER_STACK_URL

log "3/6 Unit + integration tests"
(cd "$REPO" && "$VPY" -m pytest tests/unit tests/integration -q -p no:cacheprovider -o addopts="" --timeout=300 2>&1 | tail -2) | tee "$LOGS/pytest.txt" || true
log "4/6 X3 offline: detection L0 vs L1 (test split + healthcare suite) and per-component cost (~3-5 min)"
OMP_NUM_THREADS=1 "$VPY" -u x3_offline.py --repo "$REPO" --cache "$CACHE" --sample "$SAMPLE" --out "$OUT" --core "$CORE" \
    2>&1 | grep --line-buffered -v -i "VerifyOutputSizes" | tee -a "$LOGS/x3_offline.log"
log "5/6 X2 + X3 HTTP load tests (W = 1, 2, 4 workers; L0..L3) (~25-35 min)"
"$VPY" -u x2_run.py --repo "$REPO" --sample "$SAMPLE" --out "$OUT" \
    2>&1 | grep --line-buffered -v -i "VerifyOutputSizes" | tee -a "$LOGS/x2_run.log"
log "6/6 Pack results (keys excluded)"
tar -czf results_x2.tar.gz --exclude='results/keys' results logs
log "DONE -> $HERE/results_x2.tar.gz  (copy this file back to the laptop)"
