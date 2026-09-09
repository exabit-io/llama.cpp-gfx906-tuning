# Source from every GPU benchmark script. setmax before touching the GPUs, restore at the end (and in the trap).
#   fans max (t2fanrd always_full_speed), rocm-smi perf level high, CPU governor performance, 5 s sampler,
#   host CPU RAPL PL1/PL2 = 150 W (2026-09-07: the SMC clamps the dies when the DC total passes 1228 W; four dies at the
#   200 W cap + host idle = 1062 W, + an uncapped CPU (303 W measured, 413 W allowed) = 1264 W; 150 W keeps the worst
#   case at ~1123 W with the GPU cap untouched). restore() puts the previous PL1/PL2 back.
export PATH=/opt/llama.cpp/bin:/opt/rocm/bin:$PATH
SMC=/sys/devices/LNXSYSTM:00/LNXSYBUS:00/PNP0A08:00/device:1f/APP0001:00
setmax() {
  sed -i 's/always_full_speed=false/always_full_speed=true/' /etc/t2fand.conf; systemctl restart t2fanrd
  rocm-smi --setperflevel high >/dev/null 2>&1
  cpupower frequency-set -g performance >/dev/null 2>&1
  RAPL=/sys/class/powercap/intel-rapl:0; RAPL_ORIG=$(cat $RAPL/constraint_0_power_limit_uw 2>/dev/null)
  for c in 0 1; do echo ${CPU_CAP_UW:-150000000} > $RAPL/constraint_${c}_power_limit_uw 2>/dev/null; done
  sleep 8
}
restore() {
  cpupower idle-set -E >/dev/null 2>&1
  [ -n "${RAPL_ORIG:-}" ] && for c in 0 1; do echo $RAPL_ORIG > ${RAPL:-/sys/class/powercap/intel-rapl:0}/constraint_${c}_power_limit_uw 2>/dev/null; done
  rocm-smi --setperflevel auto >/dev/null 2>&1
  cp /root/t2fand.conf.orig /etc/t2fand.conf && systemctl restart t2fanrd
  [ -n "${PROG:-}" ] && echo "$(date -Is) RESTORED perf level auto + fan curve (governor left on performance)" >> "$PROG"
}
start_sampler() {  # start_sampler FILE -> sets SAMP
  local CLK=$1; : > "$CLK"
  ( while true; do line="$(date +%H:%M:%S)"
      for d in 0b 0e 1b 1e; do dev=/sys/bus/pci/devices/0000:$d:00.0
        line="$line  $d:$(grep '\*' $dev/pp_dpm_sclk | awk '{print $2}')/$(awk '{printf "%dC", $1/1000}' $dev/hwmon/hwmon*/temp2_input | head -1)/$(awk '{printf "%dW", $1/1e6}' $dev/hwmon/hwmon*/power1_input | head -1)"; done
      echo "$line  fans:$(cat $SMC/fan1_input)/$(cat $SMC/fan2_input)/$(cat $SMC/fan3_input)/$(cat $SMC/fan4_input)" >> "$CLK"; sleep 5; done ) &
  SAMP=$!
}
state_line() { echo "kernel $(uname -r); perf: $(for d in 0b 0e 1b 1e; do cat /sys/bus/pci/devices/0000:$d:00.0/power_dpm_force_performance_level; done | tr '\n' ' '); governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor); fans rpm: $(cat $SMC/fan1_input)/$(cat $SMC/fan2_input)/$(cat $SMC/fan3_input)/$(cat $SMC/fan4_input)"; }
# wait_server PORT PID -> 0 when /health answers
wait_server() { local port=$1 pid=$2 i; for i in $(seq 1 90); do curl -sf http://127.0.0.1:$port/health >/dev/null && return 0; kill -0 $pid 2>/dev/null || return 1; sleep 5; done; return 1; }
stop_server() { kill "$@" 2>/dev/null; wait "$@" 2>/dev/null; sleep 3; }
