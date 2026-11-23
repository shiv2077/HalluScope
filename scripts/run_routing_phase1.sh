#!/usr/bin/env bash
# Background runner for routing phases 1 to 3. Resumable: every generation and label row is appended
# to its CSV as soon as it lands, so a rerun continues from the last row. Retries every 10 minutes
# for up to 8 hours if a step fails (for example an invalid OPENAI_API_KEY), re-reading .env each time.
# Usage: nohup scripts/run_routing_phase1.sh > /dev/null 2>&1 &     Log: logs/routing_phase1.log
cd "$(dirname "$0")/.."
LOG=logs/routing_phase1.log
for attempt in $(seq 1 48); do
  echo "=== attempt $attempt $(date -Is)" >> "$LOG"
  if python3 -m routing.cli run-all >> "$LOG" 2>&1; then
    echo "=== done $(date -Is)" >> "$LOG"
    exit 0
  fi
  echo "=== failed, retry in 600s" >> "$LOG"
  sleep 600
done
echo "=== gave up after 48 attempts" >> "$LOG"
exit 1
