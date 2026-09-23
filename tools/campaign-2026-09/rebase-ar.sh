#!/bin/bash
# rebase-ar.sh — re-base the AllReduce verdicts on the configuration that will actually ship.
#
# WHY. Every AR measurement so far compared custom AR against the META-BACKEND BUTTERFLY path, because
# GGML_HIP_RCCL was OFF (upstream's default, inherited without audit). With RCCL absent the init chain
# falls nccl -> internal (needs n_devices==2, we have 4) -> none -> butterfly. R3.11 now requires RCCL
# in every build, so butterfly is not the production baseline and those verdicts must be re-based.
#
# All four arms run on ONE binary via GGML_CUDA_ALLREDUCE (nccl | internal | none), so no cross-build
# comparison is needed:
#   rccl             ALLREDUCE=nccl  CUSTOM_AR=0                    the shipping default
#   rccl+customAR    ALLREDUCE=nccl  CUSTOM_AR=1 MAX_NE=20481       the production candidate
#   butterfly        ALLREDUCE=none  CUSTOM_AR=0                    what every prior run measured
#   butterfly+custAR ALLREDUCE=none  CUSTOM_AR=1 MAX_NE=20481       continuity with the old verdicts
#
# Depths follow R2.7 as corrected 2026-09-21: multi-user 4x64K; single-user 1x255K PRIMARY (same total
# KV as 4x64K, so the axes differ only in concurrency) and 1x64K CONTROL (same per-sequence depth).
# 1x32K is NOT measured: a floor is not a design point.
#
# Metrics are derived at full precision by tools/cell-metrics.py -- the tool's 2-decimal rate columns
# produce ties that break a median permutation test (2026-09-21).
set -u
W=/root/night-20260919; M=/root/models/Qwen3.8-27B-Q8_0.gguf; R=/root/rocm-tests/bench
C=/root/build-c4series-rccl
CM=/root/llama.cpp-benchmarking/tools/cell-metrics.py
log(){ echo "$(date -Is) [rebase] $*" | tee -a $W/rebase.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
[ -x "$C/bin/llama-batched-bench" ] || { echo "FATAL: $C missing" >&2; exit 1; }
strings "$C"/bin/libggml-hip.so* 2>/dev/null | grep -q ncclCommInit || { echo "FATAL: $C has no RCCL linked" >&2; exit 1; }
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
export HSA_FORCE_FINE_GRAIN_PCIE=1
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/rebase-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
rm -f $W/.rebase-done
QUEUE_PID=$$ DONEFLAG=$W/.rebase-done SMCLOG=$W/rebase-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/rebase-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/rebase-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/rebase.tsv
cell(){ # cell ARM SLOTS DEPTH REP
  local arm=$1; local s=$2; local d=$3; local rep=$4
  local out=$W/rb-$arm-$s-$d-$rep.md
  local ar cu extra=""
  case "$arm" in
    rccl)              ar=nccl; cu=0 ;;
    rccl-customAR)     ar=nccl; cu=1; extra=20481 ;;
    butterfly)         ar=none; cu=0 ;;
    butterfly-customAR) ar=none; cu=1; extra=20481 ;;
    *) log "unknown arm $arm"; return 1 ;;
  esac
  if [ -n "$extra" ]; then
    GGML_CUDA_ALLREDUCE=$ar GGML_ENABLE_CUSTOM_AR=$cu GGML_TP_AR_MAX_NE=$extra \
      timeout 3600 env LD_LIBRARY_PATH=$C/bin:$C/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib \
      $C/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
      -fa on -ctk f16 -ctv f16   `# production KV (lead 2026-09-23); q8_0 cells are superseded` -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s \
      > $out 2>$W/rb-$arm-$s-$d-$rep.log
  else
    GGML_CUDA_ALLREDUCE=$ar GGML_ENABLE_CUSTOM_AR=$cu \
      timeout 3600 env LD_LIBRARY_PATH=$C/bin:$C/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib \
      $C/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
      -fa on -ctk f16 -ctv f16   `# production KV (lead 2026-09-23); q8_0 cells are superseded` -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s \
      > $out 2>$W/rb-$arm-$s-$d-$rep.log
  fi
  local m
  if ! m=$(python3 $CM "$out" "$d" "$s" 2>>$W/rebase.progress); then
    log "CELL FAILED: $arm ${s}x$((d/1024))K rep$rep"; tail -2 $W/rb-$arm-$s-$d-$rep.log | sed 's/^/      /' | tee -a $W/rebase.progress
    printf "%s\t%s\t%s\tFAILED\tFAILED\tFAILED\n" "$arm" "$((s))x$((d/1024))K" "$rep" >> $W/rebase.tsv; return 0
  fi
  local fb; fb=$(grep -c 'falling back to meta-backend' $W/rb-$arm-$s-$d-$rep.log)
  printf "%s\t%s\t%s\t%s\t%s\n" "$arm" "$((s))x$((d/1024))K" "$rep" "$m" "$fb" >> $W/rebase.tsv
  log "$arm ${s}x$((d/1024))K rep$rep: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}'), butterfly-fallbacks=$fb"
}
log "=== re-basing AllReduce on RCCL builds. 4 arms x 3 cells x n=4 = 48 cells"
for rep in 1 2 3 4; do
  for cfg in "4 65536" "1 260864" "1 65536"; do
    set -- $cfg; s=$1; d=$2
    case $((rep % 4)) in
      1) for a in rccl rccl-customAR butterfly butterfly-customAR; do cell $a $s $d $rep; done ;;
      2) for a in rccl-customAR butterfly butterfly-customAR rccl; do cell $a $s $d $rep; done ;;
      3) for a in butterfly butterfly-customAR rccl rccl-customAR; do cell $a $s $d $rep; done ;;
      0) for a in butterfly-customAR rccl rccl-customAR butterfly; do cell $a $s $d $rep; done ;;
    esac
  done
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.rebase-done
log "=== REBASE DONE ==="
