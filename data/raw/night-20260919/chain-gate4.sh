#!/bin/bash
# chain-gate4.sh — run gate 4 once gates 1+2 are done. Chained BY PID (waitproc.sh), because a
# pattern-keyed chain is what silently skipped two gates on 2026-09-19.
set -u
W=/root/night-20260919
pid=${1:?need the gate12 pid to wait on}
/bin/bash $W/waitproc.sh "$pid" || { echo "$(date -Is) chain-gate4: waitproc failed ($?)" >> $W/chain.log; exit 1; }
echo "$(date -Is) gates 1+2 finished; starting gate 4 (graph reuse under -sm tensor)" >> $W/chain.log
sleep 20
/bin/bash $W/gate4-graphreuse.sh >> $W/gate4.out 2>&1
echo "$(date -Is) gate 4 exit=$? " >> $W/chain.log
echo "=== GATES 1-2-4 ALL DONE ===" >> $W/chain.log
