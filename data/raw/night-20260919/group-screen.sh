#!/bin/bash
# group-screen.sh — screen the substrate FEATURE GROUPS. This is the patchset binning.
#
# The substrate is a squash, so individual commits mostly cannot be removed (only 52 of 138 reverse-apply
# and 27 of those are invisible to this instrument). Feature groups are the granularity at which the
# substrate's measured +14.2% decode / +24.9% prefill over stock plausibly lives: revert every file a
# feature owns to its v0.4.1 content and measure what the substrate loses.
#
# A group that will not BUILD once reverted is itself a verdict: neutral-required-substrate, load-bearing.
# Cells at the 64K floor (R2.2 as raised 2026-09-23), f16 KV (the shipping type), against the substrate
# baseline on the same FA_QUANTS=all config.
set -u
W=/root/night-20260919; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
CM=/root/llama.cpp-benchmarking/tools/cell-metrics.py
AB=/root/llama.cpp-benchmarking/tools/assert-build-config.sh
AC=/root/llama.cpp-benchmarking/tools/assert-arms-comparable.sh
BASE=/root/build-substrate-allquants
log(){ echo "$(date -Is) [grp] $*" | tee -a $W/groupscreen.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $W/.groupscreen-done; : > $W/groupscreen.tsv
GROUPS=/tmp/claude-0/-/56cd63fc-419f-47e8-8261-04392be45e5a/scratchpad/groups.txt
[ -f "$GROUPS" ] || { echo "FATAL: group definitions missing" >&2; exit 1; }
# ---- phase 1: build every minus-group variant (host only; no measurement running)
declare -a READY=()
while IFS='|' read -r name paths; do
  [ -z "$name" ] && continue
  log "building minus-$name (reverting: $paths)"
  if /bin/bash $W/deltaminus.sh feature "$name" $paths >> $W/groupscreen-build.log 2>&1; then
    bd=/root/build-minus-$name
    if $AB "$bd" >/dev/null 2>&1; then READY+=("$name"); log "  minus-$name ready"
    else log "  minus-$name REFUSED: violates the build requirements"; fi
  else
    log "  minus-$name DOES NOT BUILD once reverted -> load-bearing, bin neutral-required-substrate"
    printf "%s\tBUILD-FAILS\t-\t-\t-\n" "$name" >> $W/groupscreen.tsv
  fi
done < "$GROUPS"
[ ${#READY[@]} -eq 0 ] && { log "nothing built; every group is load-bearing"; touch $W/.groupscreen-done; exit 0; }
$AC "$BASE" $(for g in "${READY[@]}"; do echo /root/build-minus-$g; done) >> $W/groupscreen.progress 2>&1 \
  || { log "ARMS NOT COMPARABLE — refusing to measure"; touch $W/.groupscreen-done; exit 4; }
log "arms comparable: ${#READY[@]} group(s) + the substrate baseline"
# ---- phase 2: measure
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml GGML_CUDA_ALLREDUCE=nccl
export GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash /root/rocm-tests/bench/clamp-watchdog-v2[.]sh" 2>/dev/null
  pkill -f "^/bin/bash /root/rocm-tests/bench/smc-log[.]sh" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/groupscreen-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE_PID=$$ DONEFLAG=$W/.groupscreen-done SMCLOG=$W/groupscreen-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/groupscreen-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/groupscreen-smc.log >/dev/null 2>&1 &
sleep 6
cell(){ # cell ARM BUILD REP
  local arm=$1 bld=$2 rep=$3 d=65536 s=4
  local out=$W/gs-$arm-$rep.md
  LD_LIBRARY_PATH=$bld/bin:$bld/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib timeout 5400 \
    $bld/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s \
    > $out 2>$W/gs-$arm-$rep.log
  local m
  if ! m=$(python3 $CM "$out" $d $s 2>>$W/groupscreen.progress); then
    log "CELL FAILED: $arm rep$rep"; printf "%s\tCELL-FAILED\t%s\t-\t-\n" "$arm" "$rep" >> $W/groupscreen.tsv; return 0
  fi
  printf "%s\tok\t%s\t%s\n" "$arm" "$rep" "$m" >> $W/groupscreen.tsv
  log "$arm rep$rep: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}')"
}
log "=== screening ${#READY[@]} feature groups at 4x64K, f16 KV, n=2"
for rep in 1 2; do
  cell substrate $BASE $rep
  for g in "${READY[@]}"; do cell "minus-$g" "/root/build-minus-$g" $rep; done
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash /root/rocm-tests/bench/clamp-watchdog-v2[.]sh" 2>/dev/null
pkill -f "^/bin/bash /root/rocm-tests/bench/smc-log[.]sh" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.groupscreen-done; log "=== GROUP SCREEN DONE ==="
