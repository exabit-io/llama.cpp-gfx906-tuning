#!/bin/bash
# fix-baseline.sh — build the SUBSTRATE with FA_QUANTS=all and measure the 2 cells the screen needs,
# so the 12 existing minus-build cells become interpretable. The minus data is fine; the baseline was
# wrong (it was the bundle, which carries our 20 terms = the entire apparent 2.5%).
# Then run the KV quality gate that was queued when I stopped things.
set -u
W=/root/night-20260919; WT=/root/wt-sep-test; R=/root/rocm-tests/bench
M=/root/models/Qwen3.8-27B-Q8_0.gguf; CM=/root/llama.cpp-benchmarking/tools/cell-metrics.py
AB=/root/llama.cpp-benchmarking/tools/assert-build-config.sh
BD=/root/build-substrate-allquants
log(){ echo "$(date -Is) [fix] $*" | tee -a $W/fixbase.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $W/.fixbase-done
log "building substrate with FA_QUANTS=all"
git -C $WT checkout -qf gfx906-substrate-v041 && git -C $WT clean -qfd
rm -rf $BD && mkdir -p $BD
{ cmake -S $WT -B $BD -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 \
    -DCMAKE_HIP_ARCHITECTURES=gfx906 -DGGML_HIP_RCCL=ON -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON \
    -DGGML_CUDA_FA_QUANTS=all -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
  cmake --build $BD -j 24; } > $W/build-substrate-allquants.log 2>&1
if [ ! -x $BD/bin/llama-batched-bench ] || ! $AB $BD >>$W/fixbase.progress 2>&1; then
  log "BUILD FAILED or violates config"; grep -m5 -E 'error:' $W/build-substrate-allquants.log >> $W/fixbase.progress
  touch $W/.fixbase-done; exit 1; fi
log "build OK; arms comparable check:"
/root/llama.cpp-benchmarking/tools/assert-arms-comparable.sh $BD /root/build-minus-481f684d3 >> $W/fixbase.progress 2>&1 \
  || { log "arms NOT comparable"; touch $W/.fixbase-done; exit 1; }
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
setmax; start_sampler $W/fixbase-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE_PID=$$ DONEFLAG=$W/.fixbase-done SMCLOG=$W/fixbase-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/fixbase-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/fixbase-smc.log >/dev/null 2>&1 &
sleep 6
for rep in 1 2; do
  out=$W/fb-substrate-$rep.md
  LD_LIBRARY_PATH=$BD/bin:$BD/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib timeout 3600 \
    $BD/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -c $(( 4*(65536+1280) )) -npp 65536 -ntg 1024 -npl 4 \
    > $out 2>$W/fb-substrate-$rep.log
  m=$(python3 $CM "$out" 65536 4 2>>$W/fixbase.progress) && {
    printf "substrate-allquants\t%s\t%s\n" "$rep" "$m" >> $W/screen.tsv
    log "substrate rep$rep: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}')"; }
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash /root/rocm-tests/bench/clamp-watchdog-v2[.]sh" 2>/dev/null
pkill -f "^/bin/bash /root/rocm-tests/bench/smc-log[.]sh" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.fixbase-done
log "=== BASELINE FIXED; running the KV quality gate that was interrupted"
/bin/bash $W/kv-quality2.sh >> $W/kvq.out 2>&1
log "=== kv quality exit=$? ==="
