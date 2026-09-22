#!/bin/bash
# kvsweep.sh — compile EVERY FlashAttention KV combination, then sweep them. One round, no presumptions.
#
# WHY (lead, 2026-09-22): picking a FA_QUANTS subset and then discovering a combination is missing has
# already cost this campaign twice, and worse, a missing kernel does not fail -- it converts K and V to
# f16, so a reading can look like a quantised-cache result when it is an f16-conversion result. Several
# historical KV verdicts may be exactly that: /opt/llama.cpp, /opt/llama.cpp-prod and
# /opt/llama.cpp-mxxm-fh all LACK the q8_0-q4_0 kernel, and only 1 of 3 historical quantised-V raw
# files even records which build produced it.
#
# GGML_CUDA_FA_QUANTS=all compiles the full 7x7 cross-product of q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16
# (49 combinations). GGML_CUDA_FA_ALL_QUANTS is deprecated in favour of it. iq4_nl is NOT in that type
# list, so an iq4_nl cache has no FA kernel at any setting and always converts -- worth knowing before
# anyone quotes the 2026-09-06 iq4_nl perplexity row.
#
# Arms: K held at q8_0 (R2.2's choice) while V sweeps every width, plus f16-f16 as the unquantised
# reference and q4_0-q4_0 to test whether quantising K is viable at all.
# Cells: 4x64K and 1x254K (the R2.7 design points) AND 8x32K -- the shape where the 2026-09-07 data says
# q4_0-V LOSES 3%, so old and new can be reconciled directly instead of assumed away.
# n=2 is a SCREEN; winners get fresh n=4. Every cell log is checked for an f16-conversion fallback.
set -u
W=/root/night-20260919; WT=/root/exabit-llama.cpp; M=/root/models/Qwen3.8-27B-Q8_0.gguf
R=/root/rocm-tests/bench; CM=/root/llama.cpp-benchmarking/tools/cell-metrics.py
AB=/root/llama.cpp-benchmarking/tools/assert-build-config.sh
NF=/root/llama.cpp-benchmarking/tools/assert-no-fa-fallback.sh
BD=/root/build-faq-allquants
log(){ echo "$(date -Is) [kv] $*" | tee -a $W/kvsweep.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $W/.kvsweep-done
# ---- phase 1: build with every combination
log "building with GGML_CUDA_FA_QUANTS=all (49 combinations)"
git -C $WT checkout -qf c4-series
rm -rf $BD && mkdir -p $BD
{
  cmake -S $WT -B $BD -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 \
    -DCMAKE_HIP_ARCHITECTURES=gfx906 -DGGML_HIP_RCCL=ON -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON \
    -DGGML_CUDA_FA_QUANTS=all -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
  cmake --build $BD -j 24
} > $W/build-allquants.log 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -x $BD/bin/llama-batched-bench ]; then
  log "BUILD FAILED exit=$rc"; grep -m6 -E 'error:' $W/build-allquants.log >> $W/kvsweep.progress
  touch $W/.kvsweep-done; exit 1
fi
nq=$(find $BD -name '*fattn*vec*.o' 2>/dev/null | wc -l)
log "BUILD OK -> $BD  ($nq fattn vec objects compiled)"
strings $BD/bin/libggml-hip.so* 2>/dev/null | grep -c ncclCommInit >/dev/null && log "RCCL linked"
# ---- phase 2: sweep
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
setmax; start_sampler $W/kvsweep-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE_PID=$$ DONEFLAG=$W/.kvsweep-done SMCLOG=$W/kvsweep-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/kvsweep-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/kvsweep-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/kvsweep.tsv
cell(){ # cell KTYPE VTYPE SLOTS DEPTH REP
  local kt=$1 vt=$2 s=$3 d=$4 rep=$5
  local arm="${kt}-${vt}"; local out=$W/kv-$arm-$s-$d-$rep.md
  LD_LIBRARY_PATH=$BD/bin:$BD/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib timeout 7200 \
    $BD/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk $kt -ctv $vt -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s \
    > $out 2>$W/kv-$arm-$s-$d-$rep.log
  # a cell that fell back to f16 conversion is not measuring what it claims
  if ! $NF "$W/kv-$arm-$s-$d-$rep.log" >/dev/null 2>&1; then
    log "FALLBACK DETECTED: $arm ${s}x$((d/1024))K rep$rep — NOT a $vt reading, discarding"
    printf "%s\t%sx%sK\t%s\tFALLBACK\tFALLBACK\n" "$arm" "$s" "$((d/1024))" "$rep" >> $W/kvsweep.tsv; return 0
  fi
  local m
  if ! m=$(python3 $CM "$out" "$d" "$s" 2>>$W/kvsweep.progress); then
    log "CELL FAILED: $arm ${s}x$((d/1024))K rep$rep"
    printf "%s\t%sx%sK\t%s\tFAILED\tFAILED\n" "$arm" "$s" "$((d/1024))" "$rep" >> $W/kvsweep.tsv; return 0
  fi
  printf "%s\t%sx%sK\t%s\t%s\n" "$arm" "$s" "$((d/1024))" "$rep" "$m" >> $W/kvsweep.tsv
  log "$arm ${s}x$((d/1024))K rep$rep: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}')"
}
log "=== sweeping 7 KV combinations x 3 cells x n=2"
for rep in 1 2; do
  for cfg in "4 65536" "1 260864" "8 32768"; do
    set -- $cfg; s=$1; d=$2
    for kv in "q8_0 q8_0" "q8_0 q5_1" "q8_0 q5_0" "q8_0 q4_1" "q8_0 q4_0" "f16 f16" "q4_0 q4_0"; do
      set -- $kv; cell $1 $2 $s $d $rep
    done
  done
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash /root/rocm-tests/bench/clamp-watchdog-v2[.]sh" 2>/dev/null
pkill -f "^/bin/bash /root/rocm-tests/bench/smc-log[.]sh" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.kvsweep-done; log "=== KV SWEEP DONE ==="
