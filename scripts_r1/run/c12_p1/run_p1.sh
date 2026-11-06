#!/usr/bin/env bash
# SAINT-IDS Comment 12, package P1 (part a): E0 extractor fidelity on CICIDS-2017 Friday + CICIoMT2024 inventory.
# Nothing is scored on CICIoMT2024 yet: the label map is fixed from the inventory first (part b, run_p1b.sh).
# Reuses /path/to/workdir/venv and the repo clone from the Comment 9 package (commit in EXPECTED_COMMIT).
# Usage:  cd /path/to/workdir/transfer_c12_p1 && mkdir -p logs && nohup bash run_p1.sh > logs/nohup.txt 2>&1 &
# Options (env): SKIP_E0=1, SKIP_INVENTORY=1, JOBS=4, and the paths below.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=${VPY:-$PREV/venv/bin/python}
REPO=${REPO:-$PREV/transfer_c9/saint}
DATA_C12=${DATA_C12:-$PREV/data_c12}
CICIDS_PCAP=${CICIDS_PCAP:-$DATA_C12/cicids2017_pcap/Friday-WorkingHours.pcap}
CICIDS_LABELLED=${CICIDS_LABELLED:-$DATA_C12/cicids2017_labelled}
CICIOMT_ROOT=${CICIOMT_ROOT:-$DATA_C12/CICIoMT2024}
WORK=${WORK:-$PREV/work_c12}
JOBS=${JOBS:-4}
EXPECTED=$(cat EXPECTED_COMMIT)
LOGS=$HERE/logs; OUT=$HERE/results; mkdir -p "$LOGS" "$OUT" "$WORK"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/steps.log"; }
if pgrep -f "c12_e0.py|c12_e1.py|c12_inventory.py" >/dev/null; then echo "a C12 run is already active - not starting another"; exit 1; fi

log "1/5 Checks"
[ -x "$VPY" ] || { echo "no python at $VPY (set VPY=...)"; exit 1; }
[ -d "$REPO/.git" ] || { echo "no repo clone at $REPO (set REPO=...)"; exit 1; }
[ "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED" ] || { echo "repo HEAD != $EXPECTED"; exit 1; }
sha256sum -c dpkt.sha256 || { echo "dpkt wheel incomplete - copy it again"; exit 1; }
"$VPY" -c "import dpkt" 2>/dev/null || "$VPY" -m pip install -q --no-index --no-deps ./dpkt-1.9.8-py3-none-any.whl
"$VPY" -c "import dpkt, scipy, pandas; print('dpkt', dpkt.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__)" | tee -a "$LOGS/steps.log"
{ echo "--- inputs"; ls -l "$CICIDS_PCAP" 2>&1 || true; ls -l "$CICIDS_LABELLED" 2>&1 | head -20 || true
  du -sh "$CICIOMT_ROOT" 2>&1 || true; echo "--- free disk (work dir)"; df -h "$WORK"; nproc; free -g; } | tee -a "$LOGS/steps.log"
export PYTHONDONTWRITEBYTECODE=1
unset ALERT_WEBHOOK_URL SAINT_MODELS_DIR SAINT_ALERT_POLICY CKKS_PUBLIC_CONTEXT CKKS_HMAC_SECRET DOCKER_STACK_URL

log "2/5 Self-test (seconds)"
"$VPY" -u c12_selftest.py --repo "$REPO" --tmp "$WORK/selftest" 2>&1 | tee -a "$LOGS/selftest.log"

if [ "${SKIP_E0:-0}" != 1 ]; then
  [ -f "$CICIDS_PCAP" ] || { echo "missing $CICIDS_PCAP (or set SKIP_E0=1)"; exit 1; }
  ls "$CICIDS_LABELLED"/Friday*.csv >/dev/null || { echo "no Friday*.csv in $CICIDS_LABELLED (or set SKIP_E0=1)"; exit 1; }
  log "3/5 E0: extractor fidelity, Friday PCAP, 2 passes (expect 10-60 min; progress every 500k packets)"
  "$VPY" -u c12_e0.py --repo "$REPO" --pcap "$CICIDS_PCAP" --labelled "$CICIDS_LABELLED" --work "$WORK/e0" --out "$OUT" \
      2>&1 | { grep --line-buffered -v "Column '.*' not found" || true; } | tee -a "$LOGS/e0.log"
else log "3/5 E0 skipped"; fi

if [ "${SKIP_INVENTORY:-0}" != 1 ]; then
  [ -d "$CICIOMT_ROOT" ] || { echo "missing $CICIOMT_ROOT (or set SKIP_INVENTORY=1)"; exit 1; }
  log "4/5 CICIoMT2024 inventory (sizes, SHA-256, packet counts, top talkers; $JOBS jobs)"
  "$VPY" -u c12_inventory.py --root "$CICIOMT_ROOT" --out "$OUT" --jobs "$JOBS" 2>&1 | tee -a "$LOGS/inventory.log"
else log "4/5 inventory skipped"; fi

log "5/5 Pack results (no flow tables)"
tar -czf results_c12_p1.tar.gz results logs
log "DONE -> $HERE/results_c12_p1.tar.gz  (copy this file back to the laptop)"
