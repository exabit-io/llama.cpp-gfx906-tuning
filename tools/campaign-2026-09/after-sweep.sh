#!/bin/bash
# after-sweep.sh — when the KV sweep finishes, screen the next binning batch.
# Waits on the sweep's PID rather than a flag file: flags go stale, pids do not.
set -u
W=/root/night-20260919
pid=${1:?need the kvsweep pid}
/bin/bash $W/wait-job.sh "$pid" $W/.kvsweep-done || echo "$(date -Is) [after] sweep did not complete cleanly" >> $W/after.log
echo "$(date -Is) [after] sweep done; starting the delta-minus screen batch" >> $W/after.log
/bin/bash $W/screen-batch.sh 6 >> $W/screen.out 2>&1
echo "$(date -Is) [after] screen batch exit=$?" >> $W/after.log
