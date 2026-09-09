#!/bin/bash
# Clamp watchdog v2 = clamp-watchdog.sh parametrised for follow-on queues (the v1 file is left untouched while queue-after-reboot.sh runs it).
#   QUEUE      queue script name to guard (pgrep/pkill pattern; default queue-after-reboot.sh)
#   DONEFLAG   file whose presence means the queue finished
#   SMCLOG     smc-log.sh output to read the SMC DC total (PZ0G) from
#   SCLK_GUARD 1 (default): a die busy >= 30% at <= 1000 MHz for 24 samples (2 min) is the clamp; 0: off (power-cap sweep, where a
#              lowered cap legitimately holds sclk down); the DC-envelope guard (>= 1228 W once, >= 1200 W twice) is always on.
B=/root/rocm-tests/bench; Q=$B/bench-queue.progress; FLAG=$B/.clamp-detected; THRESH=24
QUEUE=${QUEUE:-queue-after-reboot.sh}; DONEFLAG=${DONEFLAG:-$B/.queue-after-reboot-done}; SCLK_GUARD=${SCLK_GUARD:-1}
SMCLOG=${SMCLOG:-$B/smc-power-after-reboot.log}; DCENV=1228; DCWARN=1200; dccnt=0; lastts=""
declare -A cnt
echo "$(date -Is) watchdog v2 start pid $$ guarding $QUEUE (sclk guard $SCLK_GUARD; DC envelope guard: >= $DCENV W once or >= $DCWARN W twice, from $SMCLOG)"
kill_chain() {  # $1 = message
  local job; job=$(grep -E ' RUN ' $Q | tail -1 | awk '{print $NF}')
  local msg="$1 during $job; killing queue chain ($QUEUE)"
  echo "$(date -Is) $msg"; echo "$(date -Is) $msg" >> $Q; echo "$(date -Is) $job$line" > $FLAG
  pkill -f "$QUEUE"; sleep 1
  pkill -f "$B/$job" 2>/dev/null; pkill -f llama-server; pkill -f llama-bench; pkill -f llama-batched-bench; pkill -f llama-perplexity; pkill -f llama-cli; pkill -f rocprof
  sleep 3; pkill -9 -f llama- 2>/dev/null
  for dev in /sys/class/drm/card[0-9]/device; do echo auto > $dev/power_dpm_force_performance_level 2>/dev/null; done
  for dev in /sys/bus/pci/devices/0000:{0b,0e,1b,1e}:00.0; do echo 200000000 > $dev/hwmon/hwmon*/power1_cap 2>/dev/null; done
  exit 2
}
while true; do
  [ -f $DONEFLAG ] && { echo "$(date -Is) queue done, watchdog exit"; exit 0; }
  pgrep -f "$QUEUE" >/dev/null || { echo "$(date -Is) queue not running, watchdog exit"; exit 0; }
  line=""
  for dev in /sys/class/drm/card[0-9]/device; do
    [ -f $dev/gpu_busy_percent ] || continue
    d=$(basename $(dirname $dev)); busy=$(cat $dev/gpu_busy_percent 2>/dev/null || echo 0)
    mhz=$(grep '\*' $dev/pp_dpm_sclk | awk '{print $2}' | tr -dc '0-9')
    if [ "$SCLK_GUARD" = 1 ] && [ "${busy:-0}" -ge 30 ] && [ "${mhz:-9999}" -le 1000 ]; then cnt[$d]=$(( ${cnt[$d]:-0} + 1 )); else cnt[$d]=0; fi
    line="$line $d:${mhz}MHz/${busy}%/${cnt[$d]}"
    if [ ${cnt[$d]} -ge $THRESH ]; then
      kill_chain "CLAMP DETECTED by watchdog: $d busy>=30% at <=1000 MHz for $((THRESH*5)) s (needs a cold power cycle)"
    fi
  done
  last=$(tail -1 $SMCLOG 2>/dev/null); ts=${last%% *}
  dc=$(sed -n 's/.*PZ0G=\([0-9]*\).*/\1/p' <<<"$last")
  if [ -n "$dc" ] && [ "$ts" != "$lastts" ]; then
    lastts=$ts; age=$(( $(date +%s) - $(date -d "$ts" +%s 2>/dev/null || echo 0) ))
    if [ $age -ge -5 ] && [ $age -le 20 ]; then
      line="$line DC:${dc}W"
      if [ "$dc" -ge $DCENV ]; then kill_chain "DC ENVELOPE: SMC DC total ${dc} W >= ${DCENV} W envelope at $ts (the SMC clamps ~15 s later; host load beside four dies?)"; fi
      if [ "$dc" -ge $DCWARN ]; then dccnt=$((dccnt+1)); else dccnt=0; fi
      if [ $dccnt -ge 2 ]; then kill_chain "DC ENVELOPE: SMC DC total >= ${DCWARN} W in two consecutive samples (${dc} W at $ts), 28 W under the 1228 W envelope"; fi
    fi
  fi
  echo "$(date -Is)$line"
  sleep 5
done
