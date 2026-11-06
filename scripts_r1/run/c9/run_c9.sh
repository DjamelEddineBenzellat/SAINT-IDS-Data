#!/usr/bin/env bash
# SAINT-IDS Comment 9 — workstation evidence: tests, dataset regression (decisions unchanged),
# linkage-risk analysis (c11) and a short full-system HTTP re-check (W = 1) with policy 2.3.0.
# Reuses /path/to/workdir/venv, /path/to/workdir/cache, /path/to/workdir/data, the latency sample and the X2 clone.
# Usage:  cd /path/to/workdir/transfer_c9 && mkdir -p logs && nohup bash run_c9.sh > logs/nohup.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=$PREV/venv/bin/python
CACHE=${CACHE:-$PREV/cache}
DATA=${DATA:-$PREV/data/cicids2017}
SAMPLE=${SAMPLE:-$PREV/transfer_latency/results/run1/latency_sample_v2.npz}
BASE=${BASE:-$PREV/transfer_x2/saint}
X2JSON=${X2JSON:-$PREV/transfer_x2/results/x2_x3.json}
BRANCH=004-cloud-ids-aggregation-engine
EXPECTED=$(cat EXPECTED_COMMIT)
REPO=$HERE/saint
LOGS=$HERE/logs; OUT=$HERE/results; mkdir -p "$LOGS" "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps.log"; }
if pgrep -f "x2_run.py|c11_linkage.py" >/dev/null; then echo "a run is already active - not starting another"; exit 1; fi

log "1/6 Repository (commit ${EXPECTED:0:7})"
if [ ! -d "$REPO/.git" ]; then
  sha256sum -c saint_update.bundle.sha256 || { echo "update bundle is incomplete/corrupt - copy it again"; exit 1; }
  [ -d "$BASE/.git" ] || { echo "no base clone at $BASE (set BASE=...)"; exit 1; }
  git -c core.autocrlf=false clone -q "$BASE" "$REPO"
  git -C "$REPO" fetch -q "$HERE/saint_update.bundle" "$BRANCH"
  git -C "$REPO" checkout -q -B "$BRANCH" FETCH_HEAD
fi
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
unset ALERT_WEBHOOK_URL SAINT_MODELS_DIR SAINT_ALERT_POLICY CKKS_PUBLIC_CONTEXT CKKS_HMAC_SECRET IP_PSEUDONYM_KEY DOCKER_STACK_URL

log "2/6 Unit + integration tests (incl. the PHI-exposure suite)"
(cd "$REPO" && "$VPY" -m pytest tests/unit tests/integration -q -p no:cacheprovider -o addopts="" --timeout=300 2>&1 | tail -2) | tee "$LOGS/pytest.txt" || true
log "3/6 Dataset regression (decisions must be unchanged)"
(cd "$REPO" && SAINT_CICIDS_DIR="$DATA" "$VPY" -m pytest tests/regression -v -p no:cacheprovider -o addopts="" 2>&1 | tail -5) | tee "$LOGS/pytest_regression.txt" || true
log "4/6 Linkage-risk analysis on 2.43 M flows (~2-4 min)"
"$VPY" -u c11_linkage.py --repo "$REPO" --cache "$CACHE" --out "$OUT" --x2 "$X2JSON" 2>&1 | tee -a "$LOGS/c11.log"
log "5/6 Short full-system HTTP re-check, 1 worker (~4 min)"
"$VPY" -u x2_run.py --repo "$REPO" --sample "$SAMPLE" --out "$OUT/x2_recheck" --workers 1 --no-x3 \
    2>&1 | grep --line-buffered -v -i "VerifyOutputSizes" | tee -a "$LOGS/x2_recheck.log"
log "6/6 Pack results (keys excluded)"
tar -czf results_c9.tar.gz --exclude='results/x2_recheck/keys' results logs
log "DONE -> $HERE/results_c9.tar.gz  (copy this file back to the laptop)"
