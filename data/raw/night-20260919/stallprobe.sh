#!/bin/bash
# stallprobe.sh — how often does the four-die single-stream stall fire, and what n does the
# single-user axis therefore need?
#
# WHY: gate 2 (2026-09-20) measured 1x32K decode at 40.18 / 40.19 / 40.22 / 37.74. The odd one is
# not noise: prefill in that cell was normal, the loss is entirely in the TG phase (+1.65 s), and
# the clock trace catches all four dies at the 1000 MHz DPM floor at ~34 W for one sample INSIDE the
# cell, with DC at 770 W of a 1228 W envelope and PZ0T = 0. That is the intermittent four-die
# single-stream stall already recorded for the fork's tensor-parallel state, reproducing.
#
# Consequence: single-stream decode is HEAVY-TAILED (tight ~0.05% CV plus rare ~6% stalls), so a mean
# of n=4 is the wrong estimator -- one stall moves it by 1.5%, which is most of the 2% effect floor.
# This probe measures the stall RATE so the survey can choose n and an estimator on evidence rather
# than on the 0.187% CV that gate 3 measured on the FOUR-SLOT cell.
#
# Run-level replication (separate processes), never repetitions inside one process: the survey's
# statistics require independent runs, and per-request replication is pseudo-replication.
set -u
W=/root/night-20260919; M=/root/models/Qwen3.8-27B-Q8_0.gguf; R=/root/rocm-tests/bench
A=/root/build-substrate-v041; C=/root/build-c4series
N=${N:-12}
log(){ echo "$(date -Is) [stall] $*" | tee -a $W/stallprobe.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
. $R/gpu-test-env.sh
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/stall-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
rm -f $W/.stall-done
QUEUE_PID=$$ DONEFLAG=$W/.stall-done SMCLOG=$W/stall-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/stall-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/stall-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/stall.tsv
cell(){ # cell ARM BUILD REP
  local arm=$1; local bld=$2; local rep=$3
  local out=$W/stall-$arm-$rep.md
  LD_LIBRARY_PATH=$bld/bin:$bld/lib:/opt/rocm/core-10.0/lib timeout 1200 $bld/bin/llama-batched-bench \
    -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -ctk f16 -ctv f16   `# production KV (lead 2026-09-23); q8_0 cells are superseded` \
    -b 2048 -ub 2048 -c 33792 -npp 32768 -ntg 1024 -npl 1 > $out 2>$W/stall-$arm-$rep.log
  local row tg dec pp
  row=$(grep -E "^\| *32768 " $out | tail -1)
  if [ -z "${row:-}" ]; then
    log "CELL FAILED: $arm rep$rep"; printf "%s\t%s\tFAILED\tFAILED\tFAILED\n" "$arm" "$rep" >> $W/stall.tsv; return 0
  fi
  pp=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$7); print $7}')
  tg=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$8); print $8}')
  dec=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$9); print $9}')
  printf "%s\t%s\t%s\t%s\t%s\n" "$arm" "$rep" "${dec:-NA}" "${pp:-NA}" "${tg:-NA}" >> $W/stall.tsv
  log "$arm rep$rep: decode ${dec:-NA} tok/s, prefill ${pp:-NA} t/s, T_TG ${tg:-NA} s"
}
log "=== single-stream stall-rate probe, n=$N per arm, interleaved"
for rep in $(seq 1 $N); do
  if [ $((rep % 2)) -eq 1 ]; then cell Rsu $A $rep; cell RoursSU $C $rep
  else cell RoursSU $C $rep; cell Rsu $A $rep; fi
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.stall-done
log "=== STALLPROBE DONE ==="
