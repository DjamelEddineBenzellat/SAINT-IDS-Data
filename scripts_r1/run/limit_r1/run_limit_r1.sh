#!/usr/bin/env bash
# SAINT-IDS: full test suite after removing the per-source limit (about 2 min).
#   repo update (bundle) -> unit + integration tests -> dataset regression (decisions must stay
#   TP 63,699 / FP 432 / TN 300,625 / FN 44). Read-only for the models and the policy; no timing is taken.
# Usage:  cd /path/to/workdir/transfer_limit_r1 && mkdir -p logs && nohup bash run_limit_r1.sh > logs/nohup.txt 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=${VPY:-$PREV/venv/bin/python}
BASE=${BASE:-$PREV/transfer_c12_p2/saint}
DATA=${DATA:-$PREV/data/cicids2017}
BRANCH=004-cloud-ids-aggregation-engine
EXPECTED=$(cat EXPECTED_COMMIT)
REPO=$HERE/saint
LOGS=$HERE/logs; mkdir -p "$LOGS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps.log"; }
die() { log "STOPPED: $*"; exit 1; }

grep "i5-12400" /proc/cpuinfo >/dev/null || die "this must run on the workstation (Intel Core i5-12400)"

log "1/4 Inputs"
[ -x "$VPY" ] || die "venv not found at $VPY"
[ -d "$DATA" ] || die "dataset not found at $DATA"
sha256sum -c saint_update.bundle.sha256 || die "bundle incomplete - copy it again"

log "2/4 Repository (commit ${EXPECTED:0:7})"
if [ ! -d "$REPO/.git" ]; then
  [ -d "$BASE/.git" ] || die "no base clone at $BASE (set BASE=...)"
  git -c core.autocrlf=false clone -q "$BASE" "$REPO" || die "clone failed"
  git -C "$REPO" fetch -q "$HERE/saint_update.bundle" "$BRANCH" || die "bundle fetch failed"
  git -C "$REPO" checkout -q -B "$BRANCH" FETCH_HEAD || die "checkout failed"
fi
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || die "repo HEAD is not $EXPECTED"

export PYTHONDONTWRITEBYTECODE=1
unset ALERT_WEBHOOK_URL SAINT_MODELS_DIR SAINT_ALERT_POLICY CKKS_PUBLIC_CONTEXT CKKS_HMAC_SECRET IP_PSEUDONYM_KEY DOCKER_STACK_URL ORT_INTRA_OP_THREADS

log "3/4 Unit + integration tests"
(cd "$REPO" && "$VPY" -m pytest tests/unit tests/integration -q -p no:cacheprovider -o addopts="" --timeout=300) > "$LOGS/pytest.txt" 2>&1
RC_TESTS=$?
tail -1 "$LOGS/pytest.txt" | tee -a "$LOGS/steps.log"

log "4/4 Dataset regression"
(cd "$REPO" && SAINT_CICIDS_DIR="$DATA" "$VPY" -m pytest tests/regression -v -p no:cacheprovider -o addopts="") > "$LOGS/pytest_regression.txt" 2>&1
RC_REGRESSION=$?
tail -1 "$LOGS/pytest_regression.txt" | tee -a "$LOGS/steps.log"

git -C "$REPO" status --short > "$LOGS/repo_status_after_tests.txt"
{ echo "commit $EXPECTED"; echo "tests exit code $RC_TESTS"; echo "regression exit code $RC_REGRESSION"; } | tee "$LOGS/RESULT.txt"
tar -czf results_limit_r1.tar.gz logs
log "DONE -> $HERE/results_limit_r1.tar.gz  (copy this file back to the laptop)"
