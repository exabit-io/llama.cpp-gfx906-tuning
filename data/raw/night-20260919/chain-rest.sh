#!/bin/bash
# Remaining night work, re-sequenced 2026-09-19 21:2x on the lead's decisions:
#   - stay at 125 W; the 150/170 W cap sweep is CANCELLED (revisit later)
#   - design point is now 4 x 64K: the only cell of twelve clearing R3.1 at 125 W
#   - add production-fan validation: all corpus testing pinned fans at max
# Order is by decision value, with the spec-required items (R3.6, R3.9) inside it.
set -u
W=/root/night-20260919
run() { echo "$(date -Is) starting $1" >> $W/chain.log; sleep 20; shift; "$@"; echo "$(date -Is)   -> rc=$?" >> $W/chain.log; }
run "125 W service runs (R3.6), 4x64K design point + 6x64K" \
    env RUNLIST=$W/svc125w.runlist TAG=step7-svc125w GPU_CAP=125 /bin/bash $W/serve.sh
run "production-fan validation at 125 W (PWM curve, not max)" \
    env RUNLIST=$W/prodfans.runlist TAG=step9-prodfans GPU_CAP=125 /bin/bash $W/serve-prodfans.sh
run "clients=slots test (R3.1 'every request' question)" \
    env RUNLIST=$W/noqueue.runlist TAG=step6-noqueue /bin/bash $W/serve.sh
run "MTP A/B at the 4x64K design point (R3.9), rotated off/on/on/off" \
    env RUNLIST=$W/mtp.runlist TAG=step8-mtp /bin/bash $W/serve.sh
run "step 4b context compression with a sized cache" \
    env RUNLIST=$W/ctxcompress2.runlist TAG=step4b-ctxshift /bin/bash $W/serve.sh
run "fan power delta at idle" /bin/bash $W/fanpower.sh
echo "$(date -Is) === EVERYTHING COMPLETE ===" >> $W/chain.log
