#!/bin/bash
# stop-dp2-after-cell1.sh APID -- let dp-rerun-v110.sh finish its first cell (4 x 65536), then stop it before the second
# (1 x 260864) costs another 68 minutes of GPU time: the lead asked for another job on the GPUs (2026-11-28 03:12 UTC).
# APID = the pid of the running `bash accept-v110.sh`. The script sends ONE TERM to the screen.sh that this process
# starts for the second cell; screen.sh's own trap stops the benchmark, the sampler and the watchdog and restores the
# GPU settings; accept-v110.sh then logs STOPPED and prints the statistics of the finished cell; dp-rerun-v110.sh ends
# its eviction log and its reading. The second cell is run again later as tag dp3 (rule (U) of dp-rerun-v110.sh covers
# a cell that was not read).
set -u
W=/root/rebase-v110; APID=${1:?pid of accept-v110.sh}
c=$(tr '\0' ' ' < /proc/$APID/cmdline 2>/dev/null)
case "$c" in "bash $W/accept-v110.sh"*) ;; *) echo "pid $APID is not accept-v110.sh: $c"; exit 1;; esac
until grep -q 'SCREEN dp2-s4d65536 DONE' $W/screen-dp2-s4d65536.progress 2>/dev/null; do
  kill -0 $APID 2>/dev/null || { echo "$(date -Is) accept-v110.sh is gone, nothing to stop"; exit 0; }
  sleep 5
done
echo "$(date -Is) first cell done"
for i in $(seq 1 120); do
  for p in $(pgrep -P $APID -x bash); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$c" in "bash $W/screen.sh dp2-s1d260864 "*) kill -TERM $p; echo "$(date -Is) TERM sent to screen.sh pid $p: $c"; exit 0;; esac
  done
  kill -0 $APID 2>/dev/null || { echo "$(date -Is) accept-v110.sh ended by itself"; exit 0; }
  sleep 1
done
echo "$(date -Is) no second-cell screen.sh found in 120 s"; exit 1
