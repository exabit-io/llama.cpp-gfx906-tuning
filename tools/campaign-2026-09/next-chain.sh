#!/bin/bash
# next-chain.sh — rebuild the arms that had the wrong FA_QUANTS, re-measure E1's substrate arm, then
# run the first delta-minus screen batch. Sequential so builds never overlap measurement.
#
# WHY the rebuild: assert-build-config.sh, extended 2026-09-22 to check FA_QUANTS, found that
# build-substrate-v041-rccl carried the OLD list while stock and bundle carried the campaign value. All
# three E1 arms ran at -ctv q8_0 so the same kernel executed, and the effects (+14% decode / +25%
# prefill) dwarf any plausible code-layout artefact -- but an uncompiled combination does not fail
# loudly here, it silently takes a slower path, so a config mismatch is exactly the kind of thing that
# corrupts a comparison invisibly. Re-measure rather than reason about it.
set -u
W=/root/night-20260919; WT=/root/wt-sep-test; REPO=/root/exabit-llama.cpp
AB=/root/llama.cpp-benchmarking/tools/assert-build-config.sh
CM=/root/llama.cpp-benchmarking/tools/cell-metrics.py
M=/root/models/Qwen3.8-27B-Q8_0.gguf; R=/root/rocm-tests/bench
log(){ echo "$(date -Is) [next] $*" | tee -a $W/next.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
FA='f16-f16;q8_0-q8_0;q8_0-q4_0'
# ---- phase 1: rebuild the substrate arm with the campaign config
log "rebuilding substrate with the campaign FA_QUANTS"
git -C $WT checkout -qf gfx906-substrate-v041 && git -C $WT clean -qfd
bd=/root/build-substrate-campaign; rm -rf $bd && mkdir -p $bd
{
  cmake -S $WT -B $bd -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 \
    -DCMAKE_HIP_ARCHITECTURES=gfx906 -DGGML_HIP_RCCL=ON -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON \
    "-DGGML_CUDA_FA_QUANTS=$FA" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
  cmake --build $bd -j 24
} > $W/build-substrate-campaign.log 2>&1
if ! { [ -x $bd/bin/llama-batched-bench ] && $AB $bd >>$W/next.progress 2>&1; }; then
  log "substrate rebuild FAILED or violates config; aborting"; grep -m5 -E 'error:' $W/build-substrate-campaign.log >> $W/next.progress; exit 1
fi
log "substrate rebuild OK -> $bd"
# ---- phase 2: re-measure E1's substrate arm against the unchanged stock and bundle arms
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml GGML_CUDA_ALLREDUCE=nccl
export GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/next-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
rm -f $W/.next-done
QUEUE_PID=$$ DONEFLAG=$W/.next-done SMCLOG=$W/next-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/next-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/next-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/next.tsv
cell(){ local arm=$1 bld=$2 s=$3 d=$4 rep=$5; local out=$W/nx-$arm-$s-$d-$rep.md
  LD_LIBRARY_PATH=$bld/bin:$bld/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib timeout 5400 \
    $bld/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s \
    > $out 2>$W/nx-$arm-$s-$d-$rep.log
  local m
  if ! m=$(python3 $CM "$out" "$d" "$s" 2>>$W/next.progress); then
    log "CELL FAILED: $arm ${s}x$((d/1024))K rep$rep"; printf "%s\t%sx%sK\t%s\tFAILED\tFAILED\n" "$arm" "$s" "$((d/1024))" "$rep" >> $W/next.tsv; return 0; fi
  printf "%s\t%sx%sK\t%s\t%s\n" "$arm" "$s" "$((d/1024))" "$rep" "$m" >> $W/next.tsv
  log "$arm ${s}x$((d/1024))K rep$rep: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}')"
}
log "=== E1b: substrate on the campaign config, 4 reps x 2 cells"
for rep in 1 2 3 4; do
  cell substrate-campaign $bd 4 65536 $rep
  cell substrate-campaign $bd 1 260864 $rep
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.next-done
log "=== E1b DONE; starting the delta-minus screen batch"
/bin/bash $W/screen-batch.sh 6 >> $W/screen.out 2>&1
log "=== NEXT-CHAIN DONE ==="
