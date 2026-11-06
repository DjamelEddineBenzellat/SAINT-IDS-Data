#!/usr/bin/env bash
# Log thermals / load / memory / power every 5 s, fsync'd, to diagnose hard power-offs.
OUT=${1:-/path/to/SAINT/revision-audit/logs/hw_monitor.log}
while true; do
  t=$(for z in /sys/class/thermal/thermal_zone*; do printf "%s=%s " "$(cat $z/type)" "$(( $(cat $z/temp)/1000 ))"; done 2>/dev/null)
  f=$(awk '/MHz/{s+=$4;n++} END{if(n) printf "%d", s/n}' /proc/cpuinfo)
  m=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  p="ac=$(cat /sys/class/power_supply/ADP1/online 2>/dev/null) bat=$(cat /sys/class/power_supply/BAT0/capacity 2>/dev/null)% $(cat /sys/class/power_supply/BAT0/status 2>/dev/null)"
  echo "$(date '+%F %T') load=$(cut -d' ' -f1 /proc/loadavg) mhz=$f memavail_mb=$m $p $t" >> "$OUT"
  sync "$OUT"
  sleep 5
done
