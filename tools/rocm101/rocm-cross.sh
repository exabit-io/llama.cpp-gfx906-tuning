#!/bin/bash
# rocm-cross.sh -- where does ROCm 10.1's prefill loss come from: the code its compiler generates, or its runtime
# libraries? rocm-ab.sh measured (prefill tok/s, n=5): 4 x 64K 806.1 (10.0) vs 794.8 (10.1); 1 x 255K 410.2 vs 393.3.
# The same cell, the same source commit, the two crossed arms, one run each per axis (the A/B's spread per arm is
# 0.04-0.3 %, the effects are 1.4 % and 4.1 %):
#   c100-r101  compiled by ROCm 10.0's toolchain (/root/build-master-v100r), run on ROCm 10.1's libraries
#   c101-r100  compiled by ROCm 10.1's toolchain (/root/build-master-v101x: the v101n build with the library path that
#              the 10.1 HIP driver embeds rewritten to core-10.0), run on ROCm 10.0's libraries
# Reading: the arm that reproduces 10.1's number carries the cause (its compiler or its runtime).
set -u
W=/root/rocm101-migration-20261114; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
TL=/root/llama.cpp-benchmarking/tools; CM=$TL/cell-metrics.py; NF=$TL/assert-no-fa-fallback.sh
P=$W/rocm-cross.progress; TSV=$W/rocm-cross.tsv; DONE=$W/.rocm-cross-done; O=$W/rocm-cross; mkdir -p $O
log(){ echo "$(date -Is) [cross] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE
declare -A BD=([c100-r101]=/root/build-master-v100r [c101-r100]=/root/build-master-v101x)
declare -A RT=([c100-r101]=/opt/rocm/core-10.1 [c101-r100]=/opt/rocm/core-10.0)
ARMS="c100-r101 c101-r100"
for a in $ARMS; do
  bd=${BD[$a]}; rt=${RT[$a]}
  wrong=$(LD_LIBRARY_PATH=$bd/bin:$rt/lib ldd $bd/bin/libggml-hip.so $bd/bin/llama-batched-bench | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | grep -vc "^$rt/")
  n=$(LD_LIBRARY_PATH=$bd/bin:$rt/lib ldd $bd/bin/libggml-hip.so | awk '$3 ~ /^\/opt\/rocm/' | wc -l)
  [ "$wrong" = 0 ] && [ "$n" -gt 5 ] || { log "FATAL: $a resolves $wrong ROCm libraries outside $rt ($n found)"; touch $DONE; exit 1; }
  log "$a: $bd on $rt ($n ROCm libraries, hip $(LD_LIBRARY_PATH=$bd/bin:$rt/lib ldd $bd/bin/libggml-hip.so | awk '/libamdhip64/ {print $3}' | xargs readlink -f | sed 's/.*libamdhip64.so.//'))"
done
. $R/gpu-test-env.sh
export GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
BPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${BPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/rocm-cross-clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/rocm-cross-smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $W/rocm-cross-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/rocm-cross-smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
cell(){ # cell AXIS ARM
  local axis=$1 arm=$2 s=4 d=65536
  if [ "$axis" = single ]; then s=1; d=260864; fi
  local bd=${BD[$arm]} rt=${RT[$arm]} out=$O/$axis-$arm.md
  env LD_LIBRARY_PATH=$bd/bin:$rt/lib AMD_COMGR_CACHE_DIR=/root/.cache/comgr-cross-$arm timeout 7200 \
    $bd/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s \
    > $out 2>${out%.md}.log &
  BPID=$!
  wait $BPID
  BPID=""
  $NF "${out%.md}.log" >/dev/null 2>&1 || { log "FA FALLBACK in $axis $arm"; return 0; }
  grep -qiE 'NCCL init failed|probe AllReduce failed' "${out%.md}.log" && { log "RCCL NOT UP in $axis $arm"; return 0; }
  local m
  if ! m=$(python3 -I $CM "$out" $d $s 2>>$P); then log "CELL FAILED: $axis $arm"; return 0; fi
  printf "%s\t%s\t1\t%s\n" "$axis" "$arm" "$m" >> $TSV
  log "$axis $arm: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}')"
}
for axis in single multi; do for a in $ARMS; do cell $axis $a; done; done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore; touch $DONE; log "=== ROCM-CROSS DONE ==="
