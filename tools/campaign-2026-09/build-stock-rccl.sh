#!/bin/bash
# build-stock-rccl.sh — rebuild the D2 zero point with RCCL, per R3.11.
# Gate 1 measured stock v0.4.1 without RCCL, so the "+21% substrate prefill machinery" figure
# (633 butterfly substrate vs 523 butterfly stock) compared two builds on a collective we do not
# ship. The zero point has to be on the shipping configuration or it is not a zero point.
set -u
W=/root/night-20260919; wt=/root/wt-stock-v041; bd=/root/build-stock-v041-rccl
AB=/root/llama.cpp-benchmarking/tools/assert-build-config.sh
log(){ echo "$(date -Is) [stockrccl] $*" | tee -a $W/build-stock-rccl.log; }
: > $W/build-stock-rccl.log
for i in $(seq 1 720); do
  busy=$(pgrep -cf '^/root/build-[^ ]*/bin/llama-' || true)
  b2=$(pgrep -cf '^/bin/bash \./build-options.sh' || true)
  [ "${busy:-0}" -eq 0 ] && [ "${b2:-0}" -eq 0 ] && break
  sleep 30
done
log "host free; building stock v0.4.1 with RCCL"
rm -rf "$bd" && mkdir -p "$bd"
{
  cmake -S "$wt" -B "$bd" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_HIP_ARCHITECTURES=gfx906 \
    -DGGML_HIP_RCCL=ON -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON \
    "-DGGML_CUDA_FA_QUANTS=f16-f16;q8_0-q8_0;q8_0-q4_0" \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
  cmake --build "$bd" -j 24
} > $W/build-stock-v041-rccl.log 2>&1
rc=$?
if [ $rc -eq 0 ] && $AB "$bd" >> $W/build-stock-rccl.log 2>&1; then log "STOCK+RCCL BUILD OK -> $bd"
else log "FAILED exit=$rc"; grep -m5 -E 'error:' $W/build-stock-v041-rccl.log >> $W/build-stock-rccl.log; fi
log "=== DONE rc=$rc ==="; touch $W/.build-stock-rccl-done; exit $rc
