#!/bin/bash
# After step 4: the two items the requirements REQUIRE and that are cheap —
#   probe-alloc   §5.2's per-die KV / compute buffer itemisation
#   125 W ladder  R3.6 ("every result at the design point is reported at 200 W and at 125 W")
# The MTP A/B (R3.9) and the 125 W *service* run are launched by hand afterwards, because both
# need the design point that step 2b decides.
set -u
W=/root/night-20260919
while true; do
  [ -f $W/.step4-ctxshift-done ] && break
  pgrep -f '^/bin/bash /root/night-20260919/(serve|chain-step2|chain-step2b)[.]sh' >/dev/null 2>&1 || break
  sleep 30
done
[ -f $W/.step4-ctxshift-done ] || { echo "$(date -Is) step 4 ended without done flag; step 5 not started" >> $W/chain.log; exit 1; }
echo "$(date -Is) step 4 complete; starting probe-alloc (§5.2 buffer itemisation)" >> $W/chain.log
sleep 20
/bin/bash $W/probe-alloc.sh >> $W/probe.out 2>&1
echo "$(date -Is) probe-alloc exited rc=$?" >> $W/chain.log
echo "$(date -Is) starting 125 W ladder over the four PASSING cells (R3.6)" >> $W/chain.log
sleep 20
TAG=ladder-q8kv-125w GPU_CAP=125 CELLS="4:65536 6:65536 8:49152 4:131072" \
  /bin/bash $W/ladder.sh >> $W/ladder-125w.out 2>&1
echo "$(date -Is) 125 W ladder exited rc=$?" >> $W/chain.log
