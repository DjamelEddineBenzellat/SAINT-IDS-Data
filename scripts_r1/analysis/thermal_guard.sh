#!/usr/bin/env bash
# Stop process $1 if CPU package temp >= $2 °C (default 80) or turbo is re-enabled.
PID=$1; LIMIT=${2:-80}; LOG=${3:-/path/to/SAINT/revision-audit/logs/thermal_guard.log}
ZONE=$(for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = x86_pkg_temp ] && echo $z; done | head -1)
echo "$(date '+%F %T') guard start pid=$PID limit=${LIMIT}C zone=$ZONE" >> "$LOG"; sync "$LOG"
peak=0
while kill -0 "$PID" 2>/dev/null; do
  t=$(( $(cat $ZONE/temp)/1000 )); nt=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
  [ "$t" -gt "$peak" ] && peak=$t
  if [ "$t" -ge "$LIMIT" ] || [ "$nt" != "1" ]; then
    echo "$(date '+%F %T') ABORT temp=${t}C no_turbo=$nt -> SIGTERM $PID" >> "$LOG"; sync "$LOG"
    kill -TERM "$PID"; sleep 3; kill -KILL "$PID" 2>/dev/null
    break
  fi
  sleep 1
done
echo "$(date '+%F %T') guard end peak=${peak}C" >> "$LOG"; sync "$LOG"
