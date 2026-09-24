#!/bin/bash
# d1-single-depth.sh — re-measure the SINGLE-USER axis at the corrected design points.
#
# WHY (lead, 2026-09-21): the single-user cells were run at 1x32K, and 4x64K multi-user against 1x32K
# single-user is not an apples-to-apples axis comparison. The 32K figure came from R2.7's wording
# ("judged on single-stream decode at the R2.2 context floor") -- but a FLOOR is a minimum, not a
# comparison depth, and R2.2's own design point is 4 slots x 256K.
#
#   1x255K  PRIMARY. Same TOTAL KV as the 4x64K multi-user design point (256K), so the two axes differ
#           only in concurrency. 255K not 256K because n_ctx_train is 262144 and the cell needs room
#           for 1024 generated tokens: depth 260864 + 1280 headroom = 262144 exactly.
#   1x64K   CONTROL. Same PER-SEQUENCE depth as multi-user, isolating batching from depth.
#
# Until these reproduce, the `both` bins for tp-ar-size-gate, q8-repack and custom-allreduce are
# PROVISIONAL: their single-user arms were measured at a depth that is not the design point.
# Metrics are derived at full precision (tools/cell-metrics.py rationale): the tool's rate columns
# round to two decimals and the ties that creates broke a median permutation test on 2026-09-21.
set -u
W=/root/night-20260919; M=/root/models/Qwen3.8-27B-Q8_0.gguf; R=/root/rocm-tests/bench
A=/root/build-substrate-v041; C=/root/build-c4series
log(){ echo "$(date -Is) [sd] $*" | tee -a $W/d1sd.progress; }
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
setmax; start_sampler $W/d1sd-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
rm -f $W/.d1sd-done
QUEUE_PID=$$ DONEFLAG=$W/.d1sd-done SMCLOG=$W/d1sd-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/d1sd-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/d1sd-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/d1sd.tsv
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
    log "CELL FAILED: $arm ${s}x$((d/1024))K rep$rep"; tail -3 $W/d2-$arm-$s-$d-$rep.log | sed 's/^/      /' | tee -a $W/d1sd.progress
    printf "%s\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$((s))x$((d/1024))K" "$rep" >> $W/d1sd.tsv; return 0
  fi
  pp=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$7); print $7}')
  per=$(echo "$row" | awk -F'|' -v s=$s '{gsub(/ /,"",$9); printf "%.4f", $9/s}')
  printf "%s\t%s\t%s\t%s\t%s\n" "$arm" "$((s))x$((d/1024))K" "$rep" "${per:-NA}" "${pp:-NA}" >> $W/d1sd.tsv
  log "$arm ${s}x$((d/1024))K rep$rep: decode ${per:-NA} tok/s/slot, prefill ${pp:-NA} t/s"
}
log "=== single-user axis at the corrected depths: 1x255K (KV-matched) and 1x64K (depth-matched)"
for rep in 1 2 3 4; do
  for D in 260864 65536; do
    if [ $((rep % 2)) -eq 1 ]; then
      GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 cell S-gate-on    $C 1 $D $rep
      GGML_ENABLE_CUSTOM_AR=1                          cell S-gate-off   $C 1 $D $rep
      GGML_ENABLE_CUSTOM_AR=0                          cell S-AR-off     $C 1 $D $rep
      GGML_ENABLE_CUSTOM_AR=1 cell S-repack-on  $A 1 $D $rep
      GGML_ENABLE_CUSTOM_AR=1 cell S-repack-off $A 1 $D $rep --no-repack
    else
      GGML_ENABLE_CUSTOM_AR=1 cell S-repack-off $A 1 $D $rep --no-repack
      GGML_ENABLE_CUSTOM_AR=1 cell S-repack-on  $A 1 $D $rep
      GGML_ENABLE_CUSTOM_AR=0                          cell S-AR-off     $C 1 $D $rep
      GGML_ENABLE_CUSTOM_AR=1                          cell S-gate-off   $C 1 $D $rep
      GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 cell S-gate-on    $C 1 $D $rep
    fi
  done
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.d1sd-done
log "=== AR-EARNS-ITS-PLACE DONE ==="
