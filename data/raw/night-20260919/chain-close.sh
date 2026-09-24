#!/bin/bash
# CLOSING SEQUENCE, 2026-09-19 21:3x. The lead wants a natural stopping point to start the
# per-profile patch survey. Keeping only what the survey depends on:
#   svc125w   (running) the design point's chat-shape baseline at the production cap — this is the
#             number every candidate patch will be measured against.
#   prodfans  methodology validity: if 4x64K throttles under production PWM, then every survey
#             measurement taken at max fans is unrepresentative too.
#   fanpower  5 min, answers the lead's fan-power question and feeds the later cap revisit.
# DEFERRED to the survey itself (not dropped):
#   MTP A/B   R3.9 is a per-profile question — which build carries MTP is exactly what the survey
#             decides, so it belongs inside it rather than before it.
#   clients=slots, step 4b   both are requirements questions for the lead, not survey blockers.
set -u
W=/root/night-20260919
while pgrep -f "^/bin/bash $W/serve[.]sh" >/dev/null 2>&1; do sleep 30; done
echo "$(date -Is) starting production-fan validation at 125 W (PWM curve)" >> $W/chain.log
sleep 20
RUNLIST=$W/prodfans.runlist TAG=step9-prodfans GPU_CAP=125 /bin/bash $W/serve-prodfans.sh >> $W/step9.out 2>&1
echo "$(date -Is)   prodfans -> rc=$?" >> $W/chain.log
echo "$(date -Is) starting fan power delta at idle" >> $W/chain.log
sleep 20
/bin/bash $W/fanpower.sh >> $W/fanpower.out 2>&1
echo "$(date -Is)   fanpower -> rc=$?" >> $W/chain.log
echo "$(date -Is) === CAMPAIGN CLOSED — ready for the patch survey ===" >> $W/chain.log
