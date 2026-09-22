#!/bin/bash
# survey-round1.sh — four experiments back to back on the SHIPPING configuration (RCCL + gated custom
# AR), so the box is never idle between them. All cells at the R2.7-corrected depths: 4x64K multi-user
# and 1x254K single-user primary. Metrics derived at full precision; single-user summarised by median.
#
# E1  GATE 1 REDO / what is the fork actually worth?   stock-rccl vs substrate-rccl vs bundle(faq)
#     Gate 1's "+21% substrate prefill machinery" compared a butterfly substrate against a butterfly
#     stock, so the fork tile table's value is still unmeasured. This is the honest three-way.
# E2  GGML_CUDA_FORCE_MMQ                             faq vs faq-forcemmq
# E3  q8_0-K / q4_0-V capacity mode                    faq, -ctv q8_0 vs -ctv q4_0 (kernel now compiled)
# E4  q8 weight repack, re-based on RCCL               faq, default vs --no-repack
#
# NOT run: GGML_HIP_NO_VMM=OFF. It does not compile on ROCm at all -- HIP has no CUDA VMM API
# (CUresult/cuMemCreate). That is why the fork pins NO_VMM=ON. Unit closed as not-measurable.
set -u
W=/root/night-20260919; M=/root/models/Qwen3.8-27B-Q8_0.gguf; R=/root/rocm-tests/bench
CM=/root/llama.cpp-benchmarking/tools/cell-metrics.py
log(){ echo "$(date -Is) [r1] $*" | tee -a $W/round1.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
for b in /root/build-stock-v041-rccl /root/build-substrate-v041-rccl /root/build-faq /root/build-faq-forcemmq; do
  [ -x "$b/bin/llama-batched-bench" ] || { echo "FATAL: $b missing" >&2; exit 1; }
  /root/llama.cpp-benchmarking/tools/assert-build-config.sh "$b" >/dev/null || { echo "FATAL: $b violates R3.11" >&2; exit 1; }
done
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
export HSA_FORCE_FINE_GRAIN_PCIE=1
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/round1-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
rm -f $W/.round1-done
QUEUE_PID=$$ DONEFLAG=$W/.round1-done SMCLOG=$W/round1-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/round1-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/round1-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/round1.tsv
cell(){ # cell ARM BUILD SLOTS DEPTH REP [extra args...]
  local arm=$1; local bld=$2; local s=$3; local d=$4; local rep=$5; shift 5
  local out=$W/r1-$arm-$s-$d-$rep.md
  # stock has no fork AR code, so custom-AR env is meaningless there; every other arm runs the
  # shipping configuration: RCCL for the collective, custom AR gated for the narrow tensors.
  local envs=(GGML_CUDA_ALLREDUCE=nccl)
  [ "$arm" = "stock" ] || envs+=(GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481)
  env "${envs[@]}" LD_LIBRARY_PATH=$bld/bin:$bld/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib \
    timeout 5400 $bld/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk q8_0 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s "$@" \
    > $out 2>$W/r1-$arm-$s-$d-$rep.log
  local m
  if ! m=$(python3 $CM "$out" "$d" "$s" 2>>$W/round1.progress); then
    log "CELL FAILED: $arm ${s}x$((d/1024))K rep$rep"; tail -2 $W/r1-$arm-$s-$d-$rep.log | sed 's/^/      /' | tee -a $W/round1.progress
    printf "%s\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$((s))x$((d/1024))K" "$rep" >> $W/round1.tsv; return 0
  fi
  printf "%s\t%s\t%s\t%s\n" "$arm" "$((s))x$((d/1024))K" "$rep" "$m" >> $W/round1.tsv
  log "$arm ${s}x$((d/1024))K rep$rep: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}')"
}
S=/root/build-stock-v041-rccl; SUB=/root/build-substrate-v041-rccl
F=/root/build-faq; FM=/root/build-faq-forcemmq
log "=== E1: gate 1 redo on RCCL — stock vs substrate vs bundle, the fork's real worth"
for rep in 1 2 3 4; do
  for cfg in "4 65536" "1 260864"; do set -- $cfg; s=$1; d=$2
    case $((rep % 3)) in
      1) cell stock $S $s $d $rep -ctv q8_0; cell substrate $SUB $s $d $rep -ctv q8_0; cell bundle $F $s $d $rep -ctv q8_0 ;;
      2) cell substrate $SUB $s $d $rep -ctv q8_0; cell bundle $F $s $d $rep -ctv q8_0; cell stock $S $s $d $rep -ctv q8_0 ;;
      0) cell bundle $F $s $d $rep -ctv q8_0; cell stock $S $s $d $rep -ctv q8_0; cell substrate $SUB $s $d $rep -ctv q8_0 ;;
    esac
  done
done
log "=== E2: GGML_CUDA_FORCE_MMQ"
for rep in 1 2 3 4; do
  for cfg in "4 65536" "1 260864"; do set -- $cfg; s=$1; d=$2
    if [ $((rep % 2)) -eq 1 ]; then cell mmq-off $F $s $d $rep -ctv q8_0; cell mmq-on $FM $s $d $rep -ctv q8_0
    else cell mmq-on $FM $s $d $rep -ctv q8_0; cell mmq-off $F $s $d $rep -ctv q8_0; fi
  done
done
log "=== E3: q8_0-K / q4_0-V capacity mode (kernel now compiled)"
for rep in 1 2 3 4; do
  for cfg in "4 65536" "1 260864"; do set -- $cfg; s=$1; d=$2
    if [ $((rep % 2)) -eq 1 ]; then cell v-q8 $F $s $d $rep -ctv q8_0; cell v-q4 $F $s $d $rep -ctv q4_0
    else cell v-q4 $F $s $d $rep -ctv q4_0; cell v-q8 $F $s $d $rep -ctv q8_0; fi
  done
done
log "=== E4: q8 weight repack, re-based on RCCL"
for rep in 1 2 3 4; do
  for cfg in "4 65536" "1 260864"; do set -- $cfg; s=$1; d=$2
    if [ $((rep % 2)) -eq 1 ]; then cell repack-on $F $s $d $rep -ctv q8_0; cell repack-off $F $s $d $rep -ctv q8_0 --no-repack
    else cell repack-off $F $s $d $rep -ctv q8_0 --no-repack; cell repack-on $F $s $d $rep -ctv q8_0; fi
  done
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.round1-done
log "=== ROUND 1 DONE ==="
