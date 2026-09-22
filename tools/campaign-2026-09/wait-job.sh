#!/bin/bash
# wait-job.sh PID [MARKER_FILE] — wait for a job by PID, optionally also requiring a marker.
# Flag files go stale: on 2026-09-22 a waiter armed during a build phase saw the PREVIOUS run's
# .screen-done and returned immediately, so I read 12-hour-old data as fresh. A pid cannot go stale.
set -u
pid=${1:?need a pid}; marker=${2:-}
case "$pid" in ''|*[!0-9]*) echo "wait-job: '$pid' is not a pid" >&2; exit 2;; esac
while kill -0 "$pid" 2>/dev/null; do sleep 30; done
if [ -n "$marker" ] && [ ! -f "$marker" ]; then
  echo "wait-job: pid $pid exited WITHOUT writing $marker — the job did not complete" >&2; exit 3
fi
