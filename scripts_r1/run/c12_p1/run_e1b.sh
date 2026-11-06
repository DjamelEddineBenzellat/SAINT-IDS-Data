#!/usr/bin/env bash
# SAINT-IDS Comment 12, E1b P2: reference models trained WITH IoMT traffic (plan reports/comment12/e1b_plan.md).
#   1. self-test   2. extract the 50 CICIoMT2024 TRAIN captures (frozen E0d extractor, <=200k flows/capture; ~1 h, resumable)
#   3. train M1 (IoMT only) and M2 (CICIDS-2017 v2 + IoMT), 50 seeded TPE trials each (resumable SQLite studies), tau on tune,
#      evaluate M0/M1/M2 on the E1 test set and on the CICIDS v2 test split (hours; progress in logs/e1b.log)
# Nothing is written to the repository or edge/models*. Models stay in /path/to/workdir/work_c12/e1b_models.
# Usage:  cd /path/to/workdir/transfer_c12_p1 && nohup bash run_e1b.sh > logs/nohup_e1b.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=${VPY:-$PREV/venv/bin/python}
REPO=${REPO:-$PREV/transfer_c12_p2/saint}
EXPECTED=${EXPECTED:-5db2062eacc97ff00a621f266293edb6671997de}
CICIOMT_ROOT=${CICIOMT_ROOT:-$PREV/data/CICIoMT2024}
CICIDS_CACHE=${CICIDS_CACHE:-$PREV/cache}
WORK=${WORK:-$PREV/work_c12}
TRIALS=${TRIALS:-50}
LOGS=$HERE/logs; OUT=$HERE/results_e1b; mkdir -p "$LOGS" "$OUT" "$WORK/e1b_train" "$WORK/e1b_models"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps_e1b.log"; }
if pgrep -f "c12_e1.py|c12_e1b.py|c12_inventory.py" >/dev/null; then echo "a C12 run is already active - not starting another"; exit 1; fi
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED (REPO=$REPO)"; exit 1; }
ls "$CICIDS_CACHE"/prepared_*.npz >/dev/null || { echo "no prepared_*.npz in $CICIDS_CACHE (set CICIDS_CACHE=...)"; exit 1; }
[ -d "$WORK/e1" ] || { echo "no E1 test flows in $WORK/e1"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
unset SAINT_MODELS_DIR SAINT_ALERT_POLICY
{ echo "--- memory / disk at start"; free -g; df -h "$WORK"; nproc; } | tee -a "$LOGS/steps_e1b.log"
log "1/4 Self-test"
"$VPY" -u c12_selftest.py --repo "$REPO" --tmp "$WORK/selftest" 2>&1 | tee -a "$LOGS/selftest.log"
log "2/4 Extract the TRAIN captures (sequential, cached per capture; ~1 h)"
"$VPY" -u c12_e1.py extract --root "$CICIOMT_ROOT" --label-map label_map_train.json --extractor e0_extractor_config.json \
    --work "$WORK/e1b_train" --jobs 1 --max-flows 200000 2>&1 | tee -a "$LOGS/e1b_extract.log"
log "3/4 Train M1 + M2 ($TRIALS trials each) and evaluate M0/M1/M2 (hours; resumable: rerun this script)"
"$VPY" -u c12_e1b.py --repo "$REPO" --cicids-cache "$CICIDS_CACHE" --train-work "$WORK/e1b_train" --test-work "$WORK/e1" \
    --label-map-train label_map_train.json --label-map-test label_map.json --extractor e0_extractor_config.json \
    --out "$OUT" --models "$WORK/e1b_models" --trials "$TRIALS" 2>&1 \
    | { grep --line-buffered -v "Column '.*' not found" || true; } | tee -a "$LOGS/e1b.log"
log "4/4 Pack (models stay on the workstation; their training reports are included)"
cp "$WORK"/e1b_models/*_training.json "$OUT"/ 2>/dev/null || true
tar -czf results_c12_e1b.tar.gz results_e1b logs/e1b.log logs/e1b_extract.log logs/steps_e1b.log logs/selftest.log
log "DONE -> $HERE/results_c12_e1b.tar.gz  (copy this file back to the laptop)"
