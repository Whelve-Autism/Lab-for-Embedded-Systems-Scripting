#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${ROOT}/experiment.log"
CRON_LOG="${ROOT}/cron_execution.log"
echo "== crontab =="
crontab -l 2>/dev/null || echo "(empty)"
echo
echo "== cron_execution.log (tail) =="
if [[ -f "$CRON_LOG" ]]; then tail -n 40 "$CRON_LOG"; else echo "(missing)"; fi
echo
echo "== successful BATCH_END any_fail=0 =="
if [[ -f "$LOG" ]]; then
  grep 'BATCH_END' "$LOG" | grep 'any_fail=0' || true
  echo "count=$(grep 'BATCH_END' "$LOG" | grep -c 'any_fail=0' || true)"
else
  echo "(no experiment.log yet)"
fi
