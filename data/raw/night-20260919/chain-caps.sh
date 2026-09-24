#!/bin/bash
# Re-sequenced 2026-09-19 21:0x, after the lead asked where the 125 W cap came from.
# The cap question now BLOCKS the design point (at 125 W only 4x64K clears R3.1), so the cap
# sweep runs before everything else remaining.
#   150 W  needs the host RAPL-capped to ~350 W (an UNCAPPED host allows at most 131 W)
#   170 W  the 2026-09-08 study's -5% throughput point; needs the host capped to ~250 W
# then: clients=slots -> 125 W service (R3.6) -> MTP A/B (R3.9); step 4b chains off the final marker.
set -u
W=/root/night-20260919
wait_flag() { local f=$1 i=0
  while [ ! -f "$f" ]; do
    pgrep -f "^/bin/bash $W/(serve|ladder|probe-alloc)[.]sh" >/dev/null 2>&1 || { i=$((i+1)); [ $i -gt 3 ] && return 1; }
    sleep 30
  done; return 0; }
wait_flag $W/.ladder-q8kv-125w-done || echo "$(date -Is) 125W ladder ended without its flag; continuing" >> $W/chain.log
CELLS4="4:65536 6:65536 8:49152 4:131072"
for cap in 150 170; do
  echo "$(date -Is) starting ${cap} W ladder over the four passing cells" >> $W/chain.log
  sleep 20
  TAG=ladder-q8kv-${cap}w GPU_CAP=$cap CELLS="$CELLS4" /bin/bash $W/ladder.sh >> $W/ladder-${cap}w.out 2>&1
  echo "$(date -Is) ${cap} W ladder exited rc=$?" >> $W/chain.log
done
echo "$(date -Is) starting clients=slots test" >> $W/chain.log
sleep 20
RUNLIST=$W/noqueue.runlist TAG=step6-noqueue /bin/bash $W/serve.sh >> $W/step6.out 2>&1
echo "$(date -Is) clients=slots exited rc=$?" >> $W/chain.log
echo "$(date -Is) starting 125 W service run (R3.6)" >> $W/chain.log
sleep 20
RUNLIST=$W/svc125w.runlist TAG=step7-svc125w GPU_CAP=125 /bin/bash $W/serve.sh >> $W/step7.out 2>&1
echo "$(date -Is) 125 W service run exited rc=$?" >> $W/chain.log
echo "$(date -Is) starting MTP A/B (R3.9), rotated off/on/on/off" >> $W/chain.log
sleep 20
RUNLIST=$W/mtp.runlist TAG=step8-mtp /bin/bash $W/serve.sh >> $W/step8.out 2>&1
echo "$(date -Is) MTP A/B exited rc=$?" >> $W/chain.log
echo "$(date -Is) === ALL NIGHT WORK COMPLETE ===" >> $W/chain.log
