#!/bin/bash
# Wake mechanism. Every long GPU chain must be paired with one of these, launched via the Bash tool
# with run_in_background:true so that its EXIT re-invokes the model.
#   usage: wait-for.sh "<marker string>" [<logfile>]
# Why this exists: on 2026-09-20 the variance gate and the c4-series build both COMPLETED and sat
# unread for 4.5 h because the chains wrote "=== X DONE ===" to chain.log and nothing was reading it.
# A chain that sequences work correctly still needs something to hand control back.
set -u
M=${1:?need a marker}; L=${2:-/root/night-20260919/chain.log}
while ! grep -q "$M" "$L" 2>/dev/null; do
  # also bail out if nothing is running any more — a dead chain must wake me too, not hang
  pgrep -f '/root/night-20260919/.*[.]sh' >/dev/null 2>&1 || \
    { grep -q "$M" "$L" 2>/dev/null || { echo "ALERT: no chain alive and marker '$M' absent"; exit 2; }; }
  sleep 30
done
echo "MARKER REACHED: $M"; tail -4 "$L"
