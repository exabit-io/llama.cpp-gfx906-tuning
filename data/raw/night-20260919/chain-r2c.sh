#!/bin/bash
# chain-r2c.sh — start round 2 only when (0) the operator created .r2-go (lead decision on reboots, 2026-09-26), (1) round 1b's chain has exited cleanly and (2) the lead-approved kernel
# builds of session claude-fd are finished (/root/kbuild-queue-20260925.log shows "ALL DONE") and the host is quiet,
# so no host load overlaps a measurement. Installs the extended binrun/fncompat by rename only after 1b has exited.
set -u
W=/root/night-20260919; L=$W/chain-r2.log; KQ=/root/kbuild-queue-20260925.log
log(){ echo "$(date -Is) [chain-r2c] $*" | tee -a $L; }
PRED=${1:?usage: chain-r2c.sh FNCOMPAT_R1B_PID|none   (none = after a reboot: skip the PID wait)}
echo $$ > $W/chain-r2.pid
log "waiting for round 1b (fncompat pid $PRED)"
[ "$PRED" = none ] || WAIT_MAX=86400 $W/waitproc.sh "$PRED" >> $L 2>&1
if ! grep -q 'BINRUN DONE' $W/binrun-r1b.progress || grep -q 'STOPPED' $W/binrun-r1b.progress \
   || [ ! -f $W/.fncompat-r1b-done ] || grep -q 'CLAMP' $W/binrun-r1b-watchdog.out $W/fncompat-r1b-watchdog.out 2>/dev/null; then
  log "round 1b did not finish cleanly or a clamp was logged — round 2 NOT started"; exit 1
fi
log "round 1b done; waiting for the kernel build queue ($KQ: ALL DONE), at most 6 h"
t0=$(date +%s)
until grep -q 'ALL DONE' $KQ 2>/dev/null; do
  if [ $(( $(date +%s) - t0 )) -gt 21600 ]; then log "kernel build queue not ALL DONE after 6 h — round 2 NOT started"; exit 1; fi
  sleep 60
done
log "kernel build queue ALL DONE; waiting for the go marker $W/.r2-go (created by the operator after the lead decides on reboots)"
until [ -f $W/.r2-go ]; do
  if [ $(( $(date +%s) - t0 )) -gt 86400 ]; then log "no go marker after 24 h — round 2 NOT started"; exit 1; fi
  sleep 60
done
t0=$(date +%s)
log "go marker present; waiting for a quiet host (1-min load < 8 for 3 consecutive minutes)"
quiet=0
while [ $quiet -lt 3 ]; do
  l=$(cut -d' ' -f1 /proc/loadavg); if awk -v l="$l" 'BEGIN{exit !(l < 8)}'; then quiet=$((quiet+1)); else quiet=0; fi
  if [ $(( $(date +%s) - t0 )) -gt 7200 ]; then log "host not quiet 2 h after the go marker (load $l) — round 2 NOT started"; exit 1; fi
  sleep 60
done
mv $W/binrun.sh.ext $W/binrun.sh && mv $W/fncompat.sh.ext $W/fncompat.sh || { log "could not install the extended scripts"; exit 1; }
rm -f $W/binrun.sh.next $W/fncompat.sh.next
log "host quiet (load $(cut -d' ' -f1 /proc/loadavg)); extended binrun.sh / fncompat.sh installed; starting round 2"
export ROUND=r2 ARMS_FILE=$W/binrun-arms-r2.txt NMULTI=5 NSINGLE=6
bash $W/binrun.sh > $W/binrun-r2.out 2>&1 < /dev/null &
BP=$!
log "round 2 binrun pid $BP"
bash $W/fncompat.sh "$BP" > $W/fncompat-r2.out 2>&1 < /dev/null &
FP=$!
log "round 2 fncompat pid $FP"
wait "$BP"; wait "$FP"
log "round 2 chain exited"
