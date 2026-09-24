#!/bin/bash
# Last leg: 125 W service run (R3.6), then the MTP A/B (R3.9). Both are REQUIRED by the spec.
set -u
W=/root/night-20260919
while true; do
  [ -f $W/.step6-noqueue-done ] && break
  pgrep -f '^/bin/bash /root/night-20260919/(serve|ladder|probe-alloc|chain-step5|chain-noqueue)[.]sh' >/dev/null 2>&1 || break
  sleep 30
done
[ -f $W/.step6-noqueue-done ] || { echo "$(date -Is) clients=slots test ended without done flag; continuing anyway to the REQUIRED items" >> $W/chain.log; }
echo "$(date -Is) starting 125 W service run (R3.6)" >> $W/chain.log
sleep 20
RUNLIST=$W/svc125w.runlist TAG=step7-svc125w GPU_CAP=125 /bin/bash $W/serve.sh >> $W/step7.out 2>&1
echo "$(date -Is) 125 W service run exited rc=$?" >> $W/chain.log
echo "$(date -Is) starting MTP A/B (R3.9), rotated off/on/on/off" >> $W/chain.log
sleep 20
RUNLIST=$W/mtp.runlist TAG=step8-mtp /bin/bash $W/serve.sh >> $W/step8.out 2>&1
echo "$(date -Is) MTP A/B exited rc=$?" >> $W/chain.log
echo "$(date -Is) === ALL NIGHT WORK COMPLETE ===" >> $W/chain.log
