#!/bin/bash
# Re-run the context-compression step with a sized cache, so its RATE numbers are quotable.
# The refusal finding from the first pass stands regardless (hard HTTP 400, not a rate).
set -u
W=/root/night-20260919
while true; do
  grep -q 'ALL NIGHT WORK COMPLETE' $W/chain.log 2>/dev/null && break
  pgrep -f '^/bin/bash /root/night-20260919/(serve|ladder|probe-alloc|chain-)' >/dev/null 2>&1 || break
  sleep 30
done
echo "$(date -Is) starting step 4b (context compression, sized cache)" >> $W/chain.log
sleep 20
RUNLIST=$W/ctxcompress2.runlist TAG=step4b-ctxshift /bin/bash $W/serve.sh >> $W/step4b.out 2>&1
echo "$(date -Is) step 4b exited rc=$?" >> $W/chain.log
echo "$(date -Is) === EVERYTHING COMPLETE ===" >> $W/chain.log
