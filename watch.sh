#!/usr/bin/env bash
# Watchdog for the two collector jobs. Exits (which wakes Claude) on: both finished, a stall, or an error burst.
L="$(dirname "$0")/data/logs"
prev_err=$(cat "$L"/*.log 2>/dev/null | grep -cE "HTTP [0-9]+|ERROR|bad JSON|Traceback")  # count only new errors
while true; do
  sleep 60
  api_done=$(grep -c "FINISHED" "$L/api.log" 2>/dev/null); docs_done=$(grep -c "FINISHED" "$L/docs.log" 2>/dev/null)
  if [ "${api_done:-0}" -gt 0 ] && [ "${docs_done:-0}" -gt 0 ]; then
    echo "ALL DONE"; tail -2 "$L/api.log"; tail -2 "$L/docs.log"; exit 0; fi
  if [ "${api_done:-0}" -gt 0 ] && ! grep -q "state=complete" "$L/api.log"; then
    echo "API FINISHED BUT INCOMPLETE OR STOPPED"; tail -3 "$L/api.log"; exit 4; fi
  now=$(date +%s)
  for j in api docs; do
    f="$L/$j.log"; [ -f "$f" ] || continue
    grep -q "FINISHED" "$f" && continue
    age=$(( now - $(stat -c %Y "$f") ))
    if [ $age -gt 900 ]; then echo "STALL: $j log silent for $((age/60)) min"; tail -5 "$f"; exit 3; fi
  done
  err=$(cat "$L"/*.log 2>/dev/null | grep -cE "HTTP [0-9]+|ERROR|bad JSON|Traceback")
  # threshold is high because split-fetching bad pages logs many expected HTTP 500s
  if [ $((err - prev_err)) -ge 300 ]; then echo "ERROR BURST: $((err - prev_err)) new errors in the last check window"; grep -hE "HTTP [0-9]+|ERROR|bad JSON" "$L"/*.log | tail -5; exit 2; fi
  [ $(( 10#$(date +%M) % 10 )) -eq 0 ] && prev_err=$err
done
