# Source from every GPU benchmark script. setmax before touching the GPUs, restore at the end (and in the trap).
#   fans max + GPU caps pinned through the macpro-fand override (it owns the 1228 W budget), or without it
#   t2fanrd always_full_speed and DPM perf level high (sysfs), CPU governor performance, 5 s sampler,
#   host CPU RAPL PL1/PL2 = 150 W (2026-09-07: the SMC clamps the dies when the DC total passes 1228 W; four dies at the
#   200 W cap + host idle = 1062 W, + an uncapped CPU (303 W measured, 413 W allowed) = 1264 W; 150 W keeps the worst
#   case at ~1123 W with the GPU cap untouched). restore() puts the previous PL1/PL2 back.
export PATH=/opt/llama.cpp/bin:/opt/rocm/bin:$PATH
SMC=/sys/devices/LNXSYSTM:00/LNXSYBUS:00/PNP0A08:00/device:1f/APP0001:00
FAN_OV=/run/macpro-fand/override   # macpro-fand (when it owns the fans) takes "max" here; t2fanrd takes a config flip
fand_up() {  # macpro-fand owns the box when it is ENABLED; a transient restart (watchdog, Restart=always) must never
  # hand the fans to t2fanrd: starting t2fanrd stops macpro-fand for good through Conflicts= (2026-10-22 incident)
  systemctl is-enabled --quiet macpro-fand || return 1
  local i; for i in $(seq 1 15); do systemctl is-active --quiet macpro-fand && return 0; sleep 2; done
  systemctl start macpro-fand >/dev/null 2>&1; sleep 3
  systemctl is-active --quiet macpro-fand && return 0
  echo "gpu-test-env: WARNING macpro-fand is enabled but will not start; fans and caps left as they are (static-safe)" >&2
  return 0
}
# The dies' DPM performance level, through sysfs: rocm-smi did this until ROCm 10.0 and is not part of ROCm 10.1.
set_perf_level() { local f; for f in /sys/class/drm/card[0-9]/device/power_dpm_force_performance_level; do [ -w "$f" ] && echo "$1" > "$f" 2>/dev/null; done; return 0; }
setmax() {
  set_perf_level high
  cpupower frequency-set -g performance >/dev/null 2>&1
  if fand_up; then
    # macpro-fand owns the fans and the 1228 W power budget: fans to max, GPU caps pinned (BENCH_GPU_W, default the
    # production 125 W; the budget lowers a pin that would not fit), PL1 left to its budget governor
    mkdir -p ${FAN_OV%/*}; echo "max gpu=${BENCH_GPU_W:-125}" > $FAN_OV
  else
    sed -i 's/always_full_speed=false/always_full_speed=true/' /etc/t2fand.conf; systemctl restart t2fanrd
    RAPL=/sys/class/powercap/intel-rapl:0; RAPL_ORIG=$(cat $RAPL/constraint_0_power_limit_uw 2>/dev/null)
    for c in 0 1; do echo ${CPU_CAP_UW:-150000000} > $RAPL/constraint_${c}_power_limit_uw 2>/dev/null; done
  fi
  sleep 8
}
restore() {
  cpupower idle-set -E >/dev/null 2>&1
  [ -n "${RAPL_ORIG:-}" ] && for c in 0 1; do echo $RAPL_ORIG > ${RAPL:-/sys/class/powercap/intel-rapl:0}/constraint_${c}_power_limit_uw 2>/dev/null; done
  set_perf_level auto
  if [ -f $FAN_OV ] || systemctl is-enabled --quiet macpro-fand; then rm -f $FAN_OV
  else cp /root/t2fand.conf.orig /etc/t2fand.conf && systemctl restart t2fanrd; fi
  [ -n "${PROG:-}" ] && echo "$(date -Is) RESTORED perf level auto + fan curve (governor left on performance)" >> "$PROG"
}
# GPU dies are found through /sys/class/drm (the PCI numbers of the second module changed in Nov 2026)
start_sampler() {  # start_sampler FILE -> sets SAMP
  local CLK=$1; : > "$CLK"
  ( while true; do line="$(date +%H:%M:%S)"
      for dev in /sys/class/drm/card[0-9]/device; do d=$(basename "$(readlink -f $dev)" | cut -c6-7)
        line="$line  $d:$(grep '\*' $dev/pp_dpm_sclk | awk '{print $2}')/$(awk '{printf "%dC", $1/1000}' $dev/hwmon/hwmon*/temp2_input | head -1)/$(awk '{printf "%dW", $1/1e6}' $dev/hwmon/hwmon*/power1_input | head -1)"; done
      echo "$line  fans:$(cat $SMC/fan1_input)/$(cat $SMC/fan2_input)/$(cat $SMC/fan3_input)/$(cat $SMC/fan4_input)" >> "$CLK"; sleep 5; done ) &
  SAMP=$!
}
state_line() { echo "kernel $(uname -r); perf: $(for dev in /sys/class/drm/card[0-9]/device; do cat $dev/power_dpm_force_performance_level; done | tr '\n' ' '); governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor); fans rpm: $(cat $SMC/fan1_input)/$(cat $SMC/fan2_input)/$(cat $SMC/fan3_input)/$(cat $SMC/fan4_input)"; }
# wait_server PORT PID -> 0 when /health answers
wait_server() { local port=$1 pid=$2 i; for i in $(seq 1 90); do curl -sf http://127.0.0.1:$port/health >/dev/null && return 0; kill -0 $pid 2>/dev/null || return 1; sleep 5; done; return 1; }
stop_server() { kill "$@" 2>/dev/null; wait "$@" 2>/dev/null; sleep 3; }
