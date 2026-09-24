#!/bin/bash
# waitproc.sh — block until a predecessor finishes, the way chaining SHOULD be done.
#
# Learned 2026-09-20: `while pgrep -f 'test-backend-ops'; do sleep 30; done` never exited, because
# pgrep -f matches THE MONITORING SHELL'S OWN COMMAND LINE (this tool's shells are `/bin/bash -c ...`
# and the pattern text sits right there in argv). chain-gate.sh therefore never ran the perplexity
# gate, and chain-variance.sh waited on chain-gate.sh in turn. Two gates silently did nothing.
#
#   waitproc.sh PID              preferred: kill -0, immune to text matching
#   waitproc.sh - 'ANCHORED_ERE' fallback when the PID was not captured; MUST be anchored to ^ or an
#                                absolute path so it cannot match a shell that merely quotes it
# Both forms are bounded by WAIT_MAX seconds (default 4 h) so a lost predecessor cannot wedge a chain.
set -u
WAIT_MAX=${WAIT_MAX:-14400}
pid=${1:-}; pat=${2:-}
deadline=$((SECONDS + WAIT_MAX))
if [ -n "$pid" ] && [ "$pid" != "-" ]; then
  case "$pid" in ''|*[!0-9]*) echo "waitproc: '$pid' is not a pid" >&2; exit 2;; esac
  while kill -0 "$pid" 2>/dev/null; do
    [ $SECONDS -ge $deadline ] && { echo "waitproc: TIMEOUT after ${WAIT_MAX}s waiting on pid $pid" >&2; exit 3; }
    sleep 15
  done
  exit 0
fi
[ -n "$pat" ] || { echo "waitproc: need a PID or a pattern" >&2; exit 2; }
# scriptcheck: ok T4,T4b — $pat is REFUSED below unless it starts with ^ or /, and the loop is bounded by WAIT_MAX
case "$pat" in ^*|/*) : ;; *) echo "waitproc: REFUSING unanchored pattern '$pat' — anchor to ^ or /abs/path" >&2; exit 2;; esac
# scriptcheck: ok T4,T4b — anchor enforced by the case guard above; bounded by WAIT_MAX
while pgrep -f "$pat" >/dev/null 2>&1; do
  [ $SECONDS -ge $deadline ] && { echo "waitproc: TIMEOUT after ${WAIT_MAX}s waiting on '$pat'" >&2; exit 3; }
  sleep 15
done
