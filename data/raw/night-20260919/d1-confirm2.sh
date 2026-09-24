#!/bin/bash
# d1-confirm2.sh — finish the D-5 confirmation the signal interrupted at 11/40 cells on 2026-09-20.
#
# Banked and valid (same conditions, same instrument): C-gate-on n=3, C-gate-off n=3,
# C-repack-on n=3, C-repack-off n=2. This run adds the 5 missing multi-user cells and the whole
# single-user axis. The two lower-priority AR arms are dropped: custom AR is already recorded as
# structurally required-by the size gate, so binning it alone is not a decision anyone will make.
#
# n=4 is the minimum that can clear q<0.10: with n=3 there are only 20 permutation arrangements and
# the two-sided floor is exactly 0.10, which does not satisfy q<0.10. The effects here are large and
# consistent, but a verdict still waits for rep 4 rather than being upgraded on effect size.
#
# LAUNCH UNDER setsid. The previous run died to a signal at 16:19:30 because it stayed in the process
# group of a harness background-task wrapper that was later reaped. The guards were setsid'd; the JOB
# was not. A long GPU run owns its own session.
set -u
W=/root/night-20260919; M=/root/models/Qwen3.8-27B-Q8_0.gguf; R=/root/rocm-tests/bench
A=/root/build-substrate-v041; C=/root/build-c4series
log(){ echo "$(date -Is) [conf2] $*" | tee -a $W/d1conf2.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
for b in $A $C; do [ -x "$b/bin/llama-batched-bench" ] || { echo "FATAL: $b missing" >&2; exit 1; }; done
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
export HSA_FORCE_FINE_GRAIN_PCIE=1
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/d1conf2-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
rm -f $W/.d1conf2-done
QUEUE_PID=$$ DONEFLAG=$W/.d1conf2-done SMCLOG=$W/d1conf2-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/d1conf2-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/d1conf2-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/d1conf2.tsv
cell(){ # cell ARM BUILD SLOTS DEPTH REP EXTRA...
  local arm=$1; local bld=$2; local s=$3; local d=$4; local rep=$5; shift 5
  local out=$W/d2-$arm-$s-$d-$rep.md
  timeout 3600 env LD_LIBRARY_PATH=$bld/bin:$bld/lib:/opt/rocm/core-10.0/lib \
    $bld/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16   `# production KV (lead 2026-09-23); q8_0 cells are superseded` -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s "$@" \
    > $out 2>$W/d2-$arm-$s-$d-$rep.log
  local row per pp
  row=$(grep -E "^\| *$d " $out | tail -1)
  if [ -z "${row:-}" ]; then
    log "CELL FAILED: $arm ${s}x$((d/1024))K rep$rep"; tail -3 $W/d2-$arm-$s-$d-$rep.log | sed 's/^/      /' | tee -a $W/d1conf2.progress
    printf "%s\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$((s))x$((d/1024))K" "$rep" >> $W/d1conf2.tsv; return 0
  fi
  pp=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$7); print $7}')
  per=$(echo "$row" | awk -F'|' -v s=$s '{gsub(/ /,"",$9); printf "%.4f", $9/s}')
  printf "%s\t%s\t%s\t%s\t%s\n" "$arm" "$((s))x$((d/1024))K" "$rep" "${per:-NA}" "${pp:-NA}" >> $W/d1conf2.tsv
  log "$arm ${s}x$((d/1024))K rep$rep: decode ${per:-NA} tok/s/slot, prefill ${pp:-NA} t/s"
}
# The FIRST cell doubles as a state check: gate-on prefill must land near the banked 633.5 t/s. If it
# does not, the dies changed state between yesterday and today and the banked cells cannot be pooled.
log "=== finishing the multi-user arms (5 cells). First cell also checks state continuity vs 633.5 t/s"
GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 cell C-gate-on   $C 4 65536 4
GGML_ENABLE_CUSTOM_AR=1                          cell C-gate-off  $C 4 65536 4
GGML_ENABLE_CUSTOM_AR=1 cell C-repack-off $A 4 65536 3 --no-repack
GGML_ENABLE_CUSTOM_AR=1 cell C-repack-on  $A 4 65536 4
GGML_ENABLE_CUSTOM_AR=1 cell C-repack-off $A 4 65536 4 --no-repack
log "=== single-user axis, 1x32K, n=4, order rotated (summarised by MEDIAN: the stall fires ~1 run in 16)"
for rep in 1 2 3 4; do
  case $((rep % 2)) in
    1) GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 cell S-gate-on  $C 1 32768 $rep
       GGML_ENABLE_CUSTOM_AR=1                          cell S-gate-off $C 1 32768 $rep
       GGML_ENABLE_CUSTOM_AR=1 cell S-repack-on  $A 1 32768 $rep
       GGML_ENABLE_CUSTOM_AR=1 cell S-repack-off $A 1 32768 $rep --no-repack ;;
    0) GGML_ENABLE_CUSTOM_AR=1 cell S-repack-off $A 1 32768 $rep --no-repack
       GGML_ENABLE_CUSTOM_AR=1 cell S-repack-on  $A 1 32768 $rep
       GGML_ENABLE_CUSTOM_AR=1                          cell S-gate-off $C 1 32768 $rep
       GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 cell S-gate-on  $C 1 32768 $rep ;;
  esac
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.d1conf2-done
log "=== D1 CONFIRMATION PART 2 DONE ==="
