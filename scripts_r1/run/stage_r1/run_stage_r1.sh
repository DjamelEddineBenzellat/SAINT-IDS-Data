#!/usr/bin/env bash
# SAINT-IDS manuscript R1: per-stage latency of the deployed request path (about 8 min).
#   repo update (bundle) -> tests -> dataset regression -> stage latency, 3 runs of each ONNX setting
#   (deployed: 1 thread, no spinning / default: the setting of the earlier E4 run), same flows, same core.
# Read-only for the model and the policy. Close other heavy programs on the workstation first.
# Usage:  cd /path/to/workdir/transfer_stage_r1 && mkdir -p logs && nohup bash run_stage_r1.sh > logs/nohup.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=${VPY:-$PREV/venv/bin/python}
BASE=${BASE:-$PREV/transfer_c12_p2/saint}
DATA=${DATA:-$PREV/data/cicids2017}
SAMPLE=${SAMPLE:-$PREV/transfer_latency/results/run1/latency_sample_v2.npz}
CORE=${CORE:-2}; RUNS=${RUNS:-3}; RUN_TIMEOUT=${RUN_TIMEOUT:-600}
QUICK=${QUICK:-}; SKIP_TESTS=${SKIP_TESTS:-}
BRANCH=004-cloud-ids-aggregation-engine
EXPECTED=$(cat EXPECTED_COMMIT)
SAMPLE_SHA=$(cat SAMPLE_SHA256)
REPO=$HERE/saint
LOGS=$HERE/logs; OUT=$HERE/results; mkdir -p "$LOGS" "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps.log"; }
QUICKFLAG=""; [ -n "$QUICK" ] && QUICKFLAG="--quick"

if [ -z "${ALLOW_OTHER_CPU:-}" ] && ! grep "i5-12400" /proc/cpuinfo >/dev/null; then
  echo "this benchmark must run on the workstation (Intel Core i5-12400), not here"; exit 1
fi
if pgrep -f "stage_latency_r1.py|pytest|uvicorn|c12_e1.py" >/dev/null; then echo "another run is active - not starting"; exit 1; fi

log "1/6 Inputs"
[ -x "$VPY" ] || { echo "venv not found at $VPY"; exit 1; }
[ -f "$SAMPLE" ] || { echo "latency sample not found: $SAMPLE"; exit 1; }
[ "$(sha256sum "$SAMPLE" | cut -d' ' -f1)" = "$SAMPLE_SHA" ] || { echo "latency sample is not the file used by the earlier runs: $SAMPLE"; exit 1; }

log "2/6 Repository (commit ${EXPECTED:0:7})"
if [ ! -d "$REPO/.git" ]; then
  sha256sum -c saint_update.bundle.sha256 || { echo "bundle incomplete - copy it again"; exit 1; }
  [ -d "$BASE/.git" ] || { echo "no base clone at $BASE (set BASE=...)"; exit 1; }
  git -c core.autocrlf=false clone -q "$BASE" "$REPO"
  git -C "$REPO" fetch -q "$HERE/saint_update.bundle" "$BRANCH"
  git -C "$REPO" checkout -q -B "$BRANCH" FETCH_HEAD
fi
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }
{ uname -a; lscpu; uptime; cat "/sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor"; } > "$LOGS/machine_info.txt" 2>&1 || true

export PYTHONDONTWRITEBYTECODE=1
unset ALERT_WEBHOOK_URL SAINT_MODELS_DIR SAINT_ALERT_POLICY CKKS_PUBLIC_CONTEXT CKKS_HMAC_SECRET IP_PSEUDONYM_KEY DOCKER_STACK_URL ORT_INTRA_OP_THREADS

if [ -z "$SKIP_TESTS" ]; then
  log "3/6 Unit + integration tests"
  (cd "$REPO" && "$VPY" -m pytest tests/unit tests/integration -q -p no:cacheprovider -o addopts="" --timeout=300 2>&1 | tail -3) | tee "$LOGS/pytest.txt" || true
  log "4/6 Dataset regression (decisions must stay TP 63,699 / FP 432 / TN 300,625 / FN 44)"
  (cd "$REPO" && SAINT_CICIDS_DIR="$DATA" "$VPY" -m pytest tests/regression -v -p no:cacheprovider -o addopts="" 2>&1 | tail -6) | tee "$LOGS/pytest_regression.txt" || true
  git -C "$REPO" status --short > "$LOGS/repo_status_after_tests.txt"
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
sleep "${SETTLE_S:-20}"          # let the machine settle after the tests
for r in $(seq 1 "$RUNS"); do
  if [ $((r % 2)) -eq 1 ]; then order="deployed default"; else order="default deployed"; fi
  for mode in $order; do
    log "5/6 Latency run $r/$RUNS, ONNX $mode (about 1 min)"
    { uptime; ps -eo pcpu,comm --sort=-pcpu | head -6; } >> "$LOGS/load_before_runs.txt" 2>&1 || true
    timeout "$RUN_TIMEOUT" "$VPY" -u stage_latency_r1.py run --repo "$REPO" --sample "$SAMPLE" --out "$OUT" \
        --onnx "$mode" --run "$r" --core "$CORE" $QUICKFLAG 2>&1 \
        | { grep --line-buffered -v -i "VerifyOutputSizes" || true; } | tee -a "$LOGS/run_${mode}_$r.log"
  done
done

log "6/6 Summary and pack"
"$VPY" stage_latency_r1.py summary --out "$OUT" $QUICKFLAG | tee "$LOGS/summary.log"
tar -czf results_stage_r1.tar.gz results logs
log "DONE -> $HERE/results_stage_r1.tar.gz  (copy this file back to the laptop)"
