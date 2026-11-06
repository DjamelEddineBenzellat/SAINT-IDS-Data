#!/usr/bin/env bash
# SAINT-IDS Comment 12, policy 2.4.0 verification (~10 min):
#   repo update (bundle) -> unit + integration tests -> dataset regression (decisions on CICIDS must be unchanged)
#   -> E1 re-score of the CACHED CICIoMT2024 flows under policy 2.4.0 (no PCAP pass).
# Usage:  cd /path/to/workdir/transfer_c12_p2 && mkdir -p logs && nohup bash run_c12_p2.sh > logs/nohup.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=${VPY:-$PREV/venv/bin/python}
BASE=${BASE:-$PREV/transfer_c9/saint}
DATA=${DATA:-$PREV/data/cicids2017}
WORK=${WORK:-$PREV/work_c12}
BRANCH=004-cloud-ids-aggregation-engine
EXPECTED=$(cat EXPECTED_COMMIT)
REPO=$HERE/saint
LOGS=$HERE/logs; OUT=$HERE/results; mkdir -p "$LOGS" "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps.log"; }
if pgrep -f "c12_e1.py|pytest" >/dev/null; then echo "a run is already active - not starting another"; exit 1; fi
log "1/5 Repository (commit ${EXPECTED:0:7})"
if [ ! -d "$REPO/.git" ]; then
  sha256sum -c saint_update.bundle.sha256 || { echo "bundle incomplete - copy it again"; exit 1; }
  [ -d "$BASE/.git" ] || { echo "no base clone at $BASE (set BASE=...)"; exit 1; }
  git -c core.autocrlf=false clone -q "$BASE" "$REPO"
  git -C "$REPO" fetch -q "$HERE/saint_update.bundle" "$BRANCH"
  git -C "$REPO" checkout -q -B "$BRANCH" FETCH_HEAD
fi
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
unset ALERT_WEBHOOK_URL SAINT_MODELS_DIR SAINT_ALERT_POLICY CKKS_PUBLIC_CONTEXT CKKS_HMAC_SECRET IP_PSEUDONYM_KEY DOCKER_STACK_URL
log "2/5 Unit + integration tests"
(cd "$REPO" && "$VPY" -m pytest tests/unit tests/integration -q -p no:cacheprovider -o addopts="" --timeout=300 2>&1 | tail -2) | tee "$LOGS/pytest.txt" || true
log "3/5 Dataset regression (CICIDS decisions must be unchanged: TP 63,699 / FP 432 / TN 300,625 / FN 44)"
(cd "$REPO" && SAINT_CICIDS_DIR="$DATA" "$VPY" -m pytest tests/regression -v -p no:cacheprovider -o addopts="" 2>&1 | tail -5) | tee "$LOGS/pytest_regression.txt" || true
log "4/5 E1 re-score under policy 2.4.0 (cached flows; ~5 min)"
export IP_PSEUDONYM_KEY=c12-parity-only
"$VPY" -u c12_e1.py score --repo "$REPO" --label-map label_map.json --extractor e0_extractor_config.json \
    --work "$WORK/e1" --out "$OUT" --parity 1000 2>&1 | { grep --line-buffered -v "Column '.*' not found" || true; } | tee -a "$LOGS/e1_score_p240.log"
log "5/5 Pack"
tar -czf results_c12_p2.tar.gz --exclude='results/e1_decisions.csv.gz' results logs
log "DONE -> $HERE/results_c12_p2.tar.gz  (copy this file back to the laptop)"
