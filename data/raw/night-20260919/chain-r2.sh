#!/bin/bash
# chain-r2.sh — start round 2 only after round 1b's chain has exited cleanly, so no build overlaps a measurement.
# Installs the extended binrun/fncompat (revert + runtime-env arms) by rename only after 1b has exited:
# a running bash reads its script through fd 255 and must never see it change.
set -u
W=/root/night-20260919; L=$W/chain-r2.log
log(){ echo "$(date -Is) [chain-r2] $*" | tee -a $L; }
PRED=${1:?usage: chain-r2.sh FNCOMPAT_R1B_PID}
echo $$ > $W/chain-r2.pid
log "waiting for round 1b (fncompat pid $PRED)"
WAIT_MAX=86400 $W/waitproc.sh "$PRED" >> $L 2>&1
if ! grep -q 'BINRUN DONE' $W/binrun-r1b.progress || grep -q 'STOPPED' $W/binrun-r1b.progress \
   || [ ! -f $W/.fncompat-r1b-done ] || grep -q 'CLAMP' $W/binrun-r1b-watchdog.out $W/fncompat-r1b-watchdog.out 2>/dev/null; then
  log "round 1b did not finish cleanly or a clamp was logged — round 2 NOT started"; exit 1
fi
mv $W/binrun.sh.ext $W/binrun.sh && mv $W/fncompat.sh.ext $W/fncompat.sh || { log "could not install the extended scripts"; exit 1; }
rm -f $W/binrun.sh.next $W/fncompat.sh.next
log "extended binrun.sh / fncompat.sh installed; starting round 2"
export ROUND=r2 ARMS_FILE=$W/binrun-arms-r2.txt NMULTI=5 NSINGLE=6
bash $W/binrun.sh > $W/binrun-r2.out 2>&1 < /dev/null &
BP=$!
log "round 2 binrun pid $BP"
bash $W/fncompat.sh "$BP" > $W/fncompat-r2.out 2>&1 < /dev/null &
FP=$!
log "round 2 fncompat pid $FP"
wait "$BP"; wait "$FP"
log "round 2 chain exited"
