#!/bin/bash
set -u
W=/root/night-20260919
while true; do
  [ -f $W/.step2-chat-done ] && break
  pgrep -f '^/bin/bash /root/night-20260919/(serve|chain-step2)[.]sh' >/dev/null 2>&1 || break
  sleep 30
done
[ -f $W/.step2-chat-done ] || { echo "$(date -Is) step 2 ended without done flag; step2b not started" >> $W/chain.log; exit 1; }
echo "$(date -Is) step 2 complete; starting step 2b (sized prompt cache)" >> $W/chain.log
sleep 20
RUNLIST=$W/step2b.runlist TAG=step2b-cram /bin/bash $W/serve.sh >> $W/step2b.out 2>&1
echo "$(date -Is) step 2b exited rc=$?" >> $W/chain.log
# then the context-compression measurement (R2.4 TBC)
echo "$(date -Is) starting step 4 (context compression)" >> $W/chain.log
sleep 20
RUNLIST=$W/ctxcompress.runlist TAG=step4-ctxshift /bin/bash $W/serve.sh >> $W/step4.out 2>&1
echo "$(date -Is) step 4 exited rc=$?" >> $W/chain.log
