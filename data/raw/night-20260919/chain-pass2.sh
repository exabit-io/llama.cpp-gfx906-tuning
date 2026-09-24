#!/bin/bash
# Launch ladder pass 2 the moment pass 1 finishes, so the dies are never idle overnight.
# Same immutable ladder.sh, different TAG and CELLS (no editing of a running script).
set -u
W=/root/night-20260919
while true; do
  [ -f $W/.ladder-q8kv-200w-done ] && break
  pgrep -f '^/bin/bash /root/night-20260919/ladder[.]sh' >/dev/null 2>&1 || break
  sleep 30
done
if [ ! -f $W/.ladder-q8kv-200w-done ]; then
  echo "$(date -Is) pass 1 ended WITHOUT its done flag; not starting pass 2" >> $W/chain.log
  exit 1
fi
echo "$(date -Is) pass 1 complete; starting pass 2 (frontier cells)" >> $W/chain.log
sleep 20   # let restore() settle and the dies idle before re-capping
TAG=ladder-q8kv-200w-p2 CELLS="6:65536 8:49152 10:32768 12:32768" \
  /bin/bash $W/ladder.sh >> $W/ladder-p2.out 2>&1
echo "$(date -Is) pass 2 exited rc=$?" >> $W/chain.log
