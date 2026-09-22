#!/bin/bash
# d1-confirm.sh — D-5 FRESH confirmation, n=4, BOTH axes. New data only: the screen's own numbers are
# never reused for the verdict (winner's curse), per the plan's s1.10.
#
# What the screen surfaced, and what is being confirmed:
#   E3  the AR SIZE GATE (terms/06)  screen: decode -0.2%, prefill +23.9%  <- the headline
#   E2  q8 weight repack             screen: decode +5.7%, prefill +4.6%   <- a 'both' candidate
#   E1  custom AR itself, ungated    screen: decode +8.8%, prefill -19.0%  <- only viable WITH the gate
#
# Single-user cells are summarised by MEDIAN, not mean: the four-die single-stream stall fires about
# 1 run in 16 and shifts an n=4 mean by ~1.5%, which is most of the 2% materiality floor (gate 2).
set -u
W=/root/night-20260919; M=/root/models/Qwen3.8-27B-Q8_0.gguf; R=/root/rocm-tests/bench
A=/root/build-substrate-v041; C=/root/build-c4series
log(){ echo "$(date -Is) [conf] $*" | tee -a $W/d1conf.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
export HSA_FORCE_FINE_GRAIN_PCIE=1
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/d1conf-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
rm -f $W/.d1conf-done
QUEUE_PID=$$ DONEFLAG=$W/.d1conf-done SMCLOG=$W/d1conf-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/d1conf-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/d1conf-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/d1conf.tsv
cell(){ # cell ARM BUILD SLOTS DEPTH REP EXTRA...
  local arm=$1; local bld=$2; local s=$3; local d=$4; local rep=$5; shift 5
  local out=$W/dc-$arm-$s-$d-$rep.md
  timeout 3600 env LD_LIBRARY_PATH=$bld/bin:$bld/lib:/opt/rocm/core-10.0/lib \
    $bld/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s "$@" \
    > $out 2>$W/dc-$arm-$s-$d-$rep.log
  local row per pp
  row=$(grep -E "^\| *$d " $out | tail -1)
  if [ -z "${row:-}" ]; then
    log "CELL FAILED: $arm ${s}x$((d/1024))K rep$rep"; tail -3 $W/dc-$arm-$s-$d-$rep.log | sed 's/^/      /' | tee -a $W/d1conf.progress
    printf "%s\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$((s))x$((d/1024))K" "$rep" >> $W/d1conf.tsv; return 0
  fi
  pp=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$7); print $7}')
  per=$(echo "$row" | awk -F'|' -v s=$s '{gsub(/ /,"",$9); printf "%.4f", $9/s}')
  printf "%s\t%s\t%s\t%s\t%s\n" "$arm" "$((s))x$((d/1024))K" "$rep" "${per:-NA}" "${pp:-NA}" >> $W/d1conf.tsv
  log "$arm ${s}x$((d/1024))K rep$rep: decode ${per:-NA} tok/s/slot, prefill ${pp:-NA} t/s"
}
log "=== D-5 confirmation, n=4, multi-user 4x64K then single-user 1x32K, order rotated per rep"
for rep in 1 2 3 4; do
  if [ $((rep % 2)) -eq 1 ]; then
    GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 cell C-gate-on  $C 4 65536 $rep
    GGML_ENABLE_CUSTOM_AR=1                          cell C-gate-off $C 4 65536 $rep
  else
    GGML_ENABLE_CUSTOM_AR=1                          cell C-gate-off $C 4 65536 $rep
    GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 cell C-gate-on  $C 4 65536 $rep
  fi
  GGML_ENABLE_CUSTOM_AR=1 cell C-repack-on  $A 4 65536 $rep
  GGML_ENABLE_CUSTOM_AR=1 cell C-repack-off $A 4 65536 $rep --no-repack
done
log "=== single-user axis, 1x32K, n=4 (summarised by MEDIAN)"
for rep in 1 2 3 4; do
  GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 cell S-gate-on   $C 1 32768 $rep
  GGML_ENABLE_CUSTOM_AR=1                          cell S-gate-off  $C 1 32768 $rep
  GGML_ENABLE_CUSTOM_AR=1 cell S-repack-on  $A 1 32768 $rep
  GGML_ENABLE_CUSTOM_AR=1 cell S-repack-off $A 1 32768 $rep --no-repack
  GGML_ENABLE_CUSTOM_AR=0 cell S-AR-off     $A 1 32768 $rep
  GGML_ENABLE_CUSTOM_AR=1 cell S-AR-on      $A 1 32768 $rep
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.d1conf-done
log "=== D1 CONFIRMATION DONE ==="
