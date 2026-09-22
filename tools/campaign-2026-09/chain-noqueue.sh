#!/bin/bash
# Runs after the 125 W ladder (chain-step5.sh), which is R3.6-required and must not be displaced.
set -u
W=/root/night-20260919
while true; do
  [ -f $W/.ladder-q8kv-125w-done ] && break
  pgrep -f '^/bin/bash /root/night-20260919/(serve|ladder|probe-alloc|chain-step2b|chain-step5)[.]sh' >/dev/null 2>&1 || break
  sleep 30
done
[ -f $W/.ladder-q8kv-125w-done ] || { echo "$(date -Is) 125 W ladder ended without done flag; noqueue not started" >> $W/chain.log; exit 1; }
echo "$(date -Is) 125 W ladder complete; starting clients=slots test" >> $W/chain.log
sleep 20
RUNLIST=$W/noqueue.runlist TAG=step6-noqueue /bin/bash $W/serve.sh >> $W/step6.out 2>&1
echo "$(date -Is) clients=slots test exited rc=$?" >> $W/chain.log
