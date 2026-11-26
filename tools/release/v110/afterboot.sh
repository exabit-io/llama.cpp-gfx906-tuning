#!/bin/bash
# afterboot.sh -- the GPU queue of the v1.1.0 cycle that needs a healthy clock, started once after a reboot (root's
# crontab, @reboot line; the line removes itself here). Background: on 2026-11-26 at about 19:43 UTC all four dies
# latched at sclk 1000 MHz (the clamp of this box; every speed number of that evening after 19:43 is invalid, the
# correctness results are not). The clamp began within seconds of a 56-thread build starting while the dies were
# loaded, so: NOTHING else runs on the host beside this queue (no builds, no other tests).
# Before every job: perf level high at idle must read 1700 MHz or more on every die; if not, the chain stops and says
# so (a warm reboot did not clear the latch: cold power cycle, i.e. shut down and unplug for 30 s). Each job's own
# watchdog also stops it when a busy die sits at 1000 MHz for 2 minutes.
# Jobs, each with its own header and reading rule: final verification of the merge (clean speeds for the rollback
# modes), final verification of master with the fold check, MTP decode at depth for the tile threshold, the two
# design-point cells (threshold arms and v1.0.0), final verification of the candidates pool. About 8 hours.
set -u
W=/root/rebase-v110; P=$W/afterboot.progress
log(){ echo "$(date -Is) [afterboot] $*" | tee -a $P; }
( crontab -l 2>/dev/null | grep -v 'rebase-v110/afterboot.sh' | crontab - ) 2>/dev/null
[ -f $W/.afterboot-ran ] && { log "already ran once (remove $W/.afterboot-ran to run again)"; exit 0; }
touch $W/.afterboot-ran
sleep ${SETTLE:-180}
log "start: kernel $(uname -r), up $(uptime -p), macpro-fand $(systemctl is-active macpro-fand), dies $(ls -d /sys/class/drm/card[0-9]/device/pp_dpm_sclk 2>/dev/null | wc -l)"
systemctl is-active macpro-fand >/dev/null || { log "FATAL: macpro-fand is not active; not starting"; exit 1; }
[ "$(ls -d /sys/class/drm/card[0-9]/device/pp_dpm_sclk 2>/dev/null | wc -l)" = 4 ] || { log "FATAL: not four dies; not starting"; exit 1; }
clamp_check(){ # 0 when every die reads the top clock at idle under perf level high
  local d m bad=0
  for d in /sys/class/drm/card[0-9]/device; do [ -f $d/pp_dpm_sclk ] && echo high > $d/power_dpm_force_performance_level; done; sleep 5
  for d in /sys/class/drm/card[0-9]/device; do [ -f $d/pp_dpm_sclk ] || continue; m=$(grep '\*' $d/pp_dpm_sclk | awk '{print $2}' | tr -dc 0-9); [ "${m:-0}" -ge 1700 ] || bad=1; done
  for d in /sys/class/drm/card[0-9]/device; do [ -f $d/pp_dpm_sclk ] && echo auto > $d/power_dpm_force_performance_level; done
  return $bad
}
job(){ # job NAME COMMAND...
  local name=$1; shift
  pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running before $name"; exit 1; }
  clamp_check || { log "CLAMP: a die reads below 1700 MHz at idle under perf high before $name: chain stopped (cold power cycle needed)"; exit 2; }
  log "RUN $name"
  "$@" > $W/afterboot-$name.out 2>&1
  log "END $name rc=$?"
}
clamp_check || { log "CLAMP right after boot: the warm reboot did not clear the latch. Cold power cycle needed (shut down, unplug 30 s). Nothing was run."; exit 2; }
log "clock check passed: every die at the top state at idle"
job final-merge  bash $W/queue-final.sh /root/build-merge-v110 mergeb /root/wt-merge-v110
job final-master env FOLD=1 bash $W/queue-final.sh /root/build-master-v110 masterb /root/wt-master-v110
job mtp-depth    env TAG=mtp-depth-v110b bash $W/mtp-depth-v110.sh
job dp           env ARMS="old c23o tnone t88 t75" bash $W/dp-v110.sh
job final-cand   bash $W/queue-final.sh /root/build-candidates-v110 candb /root/wt-candidates-v110
log "=== AFTERBOOT DONE ==="
