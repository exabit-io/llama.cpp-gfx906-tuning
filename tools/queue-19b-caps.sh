#!/bin/bash
# Queue 19b: TODO 13 phase-separated cap sweep after queue-19 (watchdog sclk guard OFF: lowered caps hold sclk down legitimately).
# Usage: nohup /root/rocm-tests/bench/queue-19bb-caps.sh > /root/rocm-tests/bench/queue-19bb-caps.out 2>&1 &
B=/root/rocm-tests/bench; Q=$B/bench-queue.progress; RAPL=/sys/class/powercap/intel-rapl:0; ME=queue-19bb-caps.sh
q() { echo "$(date -Is) $*" >> $Q; }
setrapl() { local c; for c in 0 1; do echo $1 > $RAPL/constraint_${c}_power_limit_uw; done; }
clamp_check() { local d bad=0; for d in /sys/class/drm/card[0-3]/device; do echo high > $d/power_dpm_force_performance_level; done; sleep 5
  for d in /sys/class/drm/card[0-3]/device; do m=$(grep '\*' $d/pp_dpm_sclk | awk '{print $2}' | tr -dc 0-9); [ "${m:-0}" -ge 1700 ] || bad=1; done
  for d in /sys/class/drm/card[0-3]/device; do echo auto > $d/power_dpm_force_performance_level; done; return $bad; }
start_watchdog() { pkill -f "^/bin/bash $B/clamp-watchdog-v2.sh" 2>/dev/null; sleep 1
  QUEUE="^/bin/bash $B/$ME" DONEFLAG=$B/.queue-19b-done SMCLOG=$B/smc-power-queue19b.log SCLK_GUARD=$1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-q19b.out 2>&1 & }
runjob() { local s=$1; [ -f $B/.clamp-detected ] && { q "SKIP $s: clamp flag set"; return; }
  clamp_check || { q "CLAMP: perf-high idle sclk below 1700 MHz before $s; chain stopped (cold power cycle needed)"; touch $B/.clamp-detected; exit 2; }
  q "RUN $s"; $B/$s > $B/${s%.sh}.out 2>&1; q "END $s rc=$?"; }
while [ ! -f $B/.queue-19-done ]; do [ -f $B/.clamp-detected ] && { q "=== queue-19b: clamp flag set while waiting; NOT starting"; exit 2; }; sleep 60; done
[ -f $B/.clamp-detected ] && { q "=== queue-19b: clamp flag set; NOT starting"; exit 2; }
q "=== queue-19bb-caps start: up $(uptime -p); t2fanrd $(systemctl is-active t2fanrd)"
systemctl is-active t2fanrd >/dev/null || { dkms autoinstall -k $(uname -r) >/dev/null 2>&1; modprobe applesmc; systemctl restart t2fanrd; }
cpupower frequency-set -g performance >/dev/null 2>&1
RAPL_ORIG=$(cat $RAPL/constraint_0_power_limit_uw); setrapl 150000000
rm -f $B/.queue-19b-done $B/.qwen38-27b-cap-phases-done
clamp_check || { q "CLAMP at queue-19b start; not starting"; exit 2; }
nohup $B/smc-log.sh $B/smc-power-queue19b.log > /dev/null 2>&1 &
start_watchdog 0
runjob cap-phases.sh
pkill -f "smc-log[.]sh"; setrapl $RAPL_ORIG
q "=== queue-19bb-caps ALLDONE (host CPU RAPL back to $((RAPL_ORIG/1000000)) W)"; touch $B/.queue-19b-done
