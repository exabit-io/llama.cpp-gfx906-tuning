#!/bin/bash
# after-screen.sh — run the KV quality gate once the screen batch finishes. Waits on PID, not a flag.
set -u
W=/root/night-20260919
pid=${1:?need the screen-batch pid}
/bin/bash $W/wait-job.sh "$pid" || echo "$(date -Is) [after2] screen did not exit cleanly" >> $W/after.log
echo "$(date -Is) [after2] screen done; starting the KV quality gate" >> $W/after.log
/bin/bash $W/kv-quality2.sh >> $W/kvq.out 2>&1
echo "$(date -Is) [after2] kv quality exit=$?" >> $W/after.log
