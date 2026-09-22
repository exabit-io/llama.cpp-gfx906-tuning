#!/bin/bash
# Start step 2 (chat-shape validation) as soon as ladder pass 2 finishes.
set -u
W=/root/night-20260919
while true; do
  [ -f $W/.ladder-q8kv-200w-p2-done ] && break
  pgrep -f '^/bin/bash /root/night-20260919/(ladder|chain-pass2)[.]sh' >/dev/null 2>&1 || break
  sleep 30
done
if [ ! -f $W/.ladder-q8kv-200w-p2-done ]; then
  echo "$(date -Is) pass 2 ended WITHOUT its done flag; step 2 not started" >> $W/chain.log; exit 1; fi
echo "$(date -Is) pass 2 complete; starting step 2 (chat shape at 3 passing configs)" >> $W/chain.log
sleep 20
RUNLIST=$W/step2.runlist TAG=step2-chat /bin/bash $W/serve.sh >> $W/step2.out 2>&1
echo "$(date -Is) step 2 exited rc=$?" >> $W/chain.log
