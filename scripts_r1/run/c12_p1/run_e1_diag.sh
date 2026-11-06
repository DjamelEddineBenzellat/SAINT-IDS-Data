#!/usr/bin/env bash
# SAINT-IDS Comment 12, E1 diagnostic (~2 min): flow directions vs labelling, ML naming, evasion-rule statistics.
# Reads the cached E1 flow tables and results_b/e1_decisions.csv.gz; no PCAP pass, no scoring.
# Usage:  cd /path/to/workdir/transfer_c12_p1 && nohup bash run_e1_diag.sh > logs/nohup_e1_diag.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
PREV=${PREV:-$HOME/transfer}
VPY=${VPY:-$PREV/venv/bin/python}
WORK=${WORK:-$PREV/work_c12}
LOGS=$HERE/logs; OUT=$HERE/results_e1_diag; mkdir -p "$LOGS" "$OUT"
[ -f results_b/e1_decisions.csv.gz ] || { echo "missing results_b/e1_decisions.csv.gz (run_p1b.sh first)"; exit 1; }
"$VPY" -u c12_e1_diag.py --work "$WORK/e1" --label-map label_map.json --decisions results_b/e1_decisions.csv.gz \
    --out "$OUT" --friday "$WORK/e0c/e0c_flows_fin_existing.pkl" 2>&1 | tee "$LOGS/e1_diag.log"
tar -czf results_c12_e1_diag.tar.gz results_e1_diag logs/e1_diag.log
echo "DONE -> $HERE/results_c12_e1_diag.tar.gz  (copy this file back to the laptop)"
