#!/bin/bash
# build-options.sh — correct GGML_CUDA_FA_QUANTS and build the two new option units.
#
# Waits for the in-flight measurement to finish: host build load perturbs benchmark timing, and every
# cell in this campaign was measured on an idle host.
#
# 1. FA_QUANTS. Our builds compiled q4_0-q4_0 (a 4-bit K cache nobody wants) and bf16-bf16, and OMITTED
#    q8_0-q4_0 -- the combination R2.2 names as the route to 256K. The fork's own recipe in this repo has
#    had it right all along: "f16-f16;q8_0-q8_0;q8_0-q4_0". bf16 is dead weight twice over here: gfx906
#    has no native bf16 (fast_bf16_hardware_available needs RDNA3+ or CDNA) and the W-3275M host is
#    Cascade Lake-SP with no AVX512-BF16. It is nothing to do with MKL -- GGML_BLAS is OFF entirely.
#    CONSEQUENCE WORTH TESTING: CLAUDE.md records a 4-bit V cache as "non-functional here". An uncompiled
#    FA kernel looks exactly like that from outside, so that verdict may be a build artifact and a
#    capacity option may have been written off for the wrong reason.
# 2. GGML_CUDA_FORCE_MMQ=ON  — forces MMQ over hipBLAS for quantised mat-mul. A prefill lever, untested
#    here. (FORCE_CUBLAS is a recorded loser; that says nothing about this, the opposite switch.)
# 3. GGML_HIP_NO_VMM=OFF    — enables the VMM pool instead of the legacy allocator. Relevant to R3.2
#    headroom at 255K. The fork deliberately keeps NO_VMM=ON, so this is a falsification test of their
#    choice, not an assumption that VMM is better.
set -u
W=/root/night-20260919; WT=/root/exabit-llama.cpp
AB=/root/llama.cpp-benchmarking/tools/assert-build-config.sh
log(){ echo "$(date -Is) [opts] $*" | tee -a $W/build-options.log; }
: > $W/build-options.log
log "waiting for the measurement to finish (.rebase-done)"
for i in $(seq 1 720); do [ -f $W/.rebase-done ] && break; sleep 30; done
[ -f $W/.rebase-done ] || { log "TIMEOUT waiting for the measurement; not building"; exit 2; }
busy=$(pgrep -cf '^/root/build-[^ ]*/bin/llama-' || true)
[ "${busy:-0}" -gt 0 ] && { log "REFUSING: a measurement is still running"; exit 3; }
log "host is free; building"
FA='all'   # R3.11: every K/V combination
build(){ # build NAME extra-cmake-args...
  local name=$1; shift
  local bd=/root/build-$name
  rm -rf "$bd" && mkdir -p "$bd"
  {
    cmake -S "$WT" -B "$bd" -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_HIP_ARCHITECTURES=gfx906 \
      -DGGML_HIP_RCCL=ON -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON \
      "-DGGML_CUDA_FA_QUANTS=all" \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_HIP_COMPILER_LAUNCHER=ccache "$@"
    cmake --build "$bd" -j 24
  } > $W/build-$name.log 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -x "$bd/bin/llama-batched-bench" ]; then
    if ! $AB "$bd" >> $W/build-options.log 2>&1; then log "$name: FAILS the build-config assertion"; return 1; fi
    local fa; fa=$(find "$bd" -name '*fattn*vec*q*.o' 2>/dev/null | grep -oE 'q8_0-q4_0|q8_0-q8_0|q4_0-q4_0' | sort -u | tr '\n' ' ')
    log "$name: BUILD OK -> $bd   FA combos: ${fa:-none found}"
  else
    log "$name: BUILD FAILED exit=$rc"; grep -m5 -E 'error:' $W/build-$name.log >> $W/build-options.log; return 1
  fi
}
rc=0
git -C "$WT" checkout -qf c4-series 2>>$W/build-options.log
build faq                                        || rc=1   # FA_QUANTS fix alone
build faq-forcemmq -DGGML_CUDA_FORCE_MMQ=ON      || rc=1
build faq-vmm      -DGGML_HIP_NO_VMM=OFF         || rc=1
log "=== BUILD-OPTIONS DONE rc=$rc ==="
touch $W/.build-options-done
exit $rc
