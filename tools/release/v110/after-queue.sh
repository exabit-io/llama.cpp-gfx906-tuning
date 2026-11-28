#!/bin/bash
# after-queue.sh QUEUEPID -- runs after afterboot.sh (the timing queue) has ended by itself: the cut-point builds
# (attr8-build.sh, with the dies idle) and then the attribution session (attr8.sh, which holds the reading rule).
# It never runs beside the queue: it waits for QUEUEPID to be gone and for "AFTERBOOT DONE" in the queue's log; a queue
# that stopped for another reason (clock latch, FATAL) is left as it is and nothing is started.
set -u
W=/root/rebase-v110; QP=$1; P=$W/after-queue.progress
log(){ echo "$(date -Is) [after-queue] $*" | tee -a $P; }
log "waiting for the queue (pid $QP)"
while grep -qs afterboot.sh /proc/"$QP"/cmdline; do sleep 60; done
grep -q 'AFTERBOOT DONE' $W/afterboot.progress || { log "the queue ended without AFTERBOOT DONE: $(tail -1 $W/afterboot.progress | cut -c1-160) -- nothing started"; log "=== AFTER-QUEUE END (not run) ==="; exit 1; }
sleep 30
for p in llama-server llama-batched-b llama-perplexit cmake; do pgrep -x $p >/dev/null && { log "FATAL: $p is running -- nothing started"; log "=== AFTER-QUEUE END (not run) ==="; exit 1; }; done
clamp_check(){ # 0 when every die reads the top clock at idle under perf level high (as afterboot.sh)
  local d m bad=0
  for d in /sys/class/drm/card[0-9]/device; do [ -f $d/pp_dpm_sclk ] && echo high > $d/power_dpm_force_performance_level; done; sleep 5
  for d in /sys/class/drm/card[0-9]/device; do [ -f $d/pp_dpm_sclk ] || continue; m=$(grep '\*' $d/pp_dpm_sclk | awk '{print $2}' | tr -dc 0-9); [ "${m:-0}" -ge 1700 ] || bad=1; done
  for d in /sys/class/drm/card[0-9]/device; do [ -f $d/pp_dpm_sclk ] && echo auto > $d/power_dpm_force_performance_level; done
  return $bad
}
clamp_check || { log "CLAMP: a die reads below 1700 MHz at idle under perf high -- nothing started"; log "=== AFTER-QUEUE END (not run) ==="; exit 2; }
log "RUN attr8-build"
bash $W/attr8-build.sh > $W/after-queue-attr8-build.out 2>&1
log "END attr8-build rc=$? (a failed cut is left out of the session)"
clamp_check || { log "CLAMP after the builds -- session not started"; log "=== AFTER-QUEUE END (not run) ==="; exit 2; }
log "RUN attr8"
bash $W/attr8.sh > $W/after-queue-attr8.out 2>&1
log "END attr8 rc=$?"
log "=== AFTER-QUEUE END ==="
