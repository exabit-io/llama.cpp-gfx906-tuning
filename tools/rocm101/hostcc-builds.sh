#!/bin/bash
# hostcc-builds.sh -- the host-COMPILER variants of mx-llama.cpp master 290fc642a for hostcc-screen.sh. The flag screen
# (flag-builds.sh, report section 5.2) varied gcc 13's flags and left open what another host compiler does. Every variant
# here is the reference configuration (build-rocm.sh, ROCm 10.1 toolchain for the GPU code, GGML_NATIVE=ON, no host
# flags, CMake Release = -O3) with only the host C/C++ compiler changed; the GPU library is not taken from these builds.
# Existing reference: /root/build-v101-t90 (gcc 13.3, OpenMP through libgomp).
#   gccnoomp   gcc 13.3 with GGML_OPENMP=OFF. Control: the box has no OpenMP runtime headers for clang 19, so the clang
#              arms are built without OpenMP; this arm is gcc in that same configuration (only libggml-cpu differs
#              from the reference).
#   c19        Ubuntu's clang 19 (/usr/bin/clang-19, libstdc++ 13), GGML_OPENMP=OFF
#   c24        ROCm 10.1's clang 24 (/opt/rocm/core-10.1/lib/llvm/bin/clang), GGML_OPENMP=OFF
set -u
W=/root/rocm101-migration-20261114; SRC=/root/wt-master-t90; RC=/opt/rocm/core-10.1
P=$W/hostcc-builds.progress
log(){ echo "$(date -Is) [hostcc] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
[ "$(git -C $SRC rev-parse --short HEAD)" = 290fc642a ] && [ -z "$(git -C $SRC status --porcelain --untracked-files=no)" ] || { log "FATAL: source is not a clean 290fc642a"; exit 1; }
one(){ # one NAME  (EXTRA from the caller's environment)
  local n=$1 b=/root/build-hostcc-$1 rc
  bash $W/build-rocm.sh $SRC $b $W/build-hostcc-$n.log $RC > /dev/null 2>&1; rc=$?
  log "$n rc=$rc: $(grep -E '^BUILD rc=' $W/build-hostcc-$n.log | tail -1 | cut -c1-80); C $(grep -E '^CMAKE_C_COMPILER:' $b/CMakeCache.txt | cut -d= -f2), C++ $(grep -E '^CMAKE_CXX_COMPILER:' $b/CMakeCache.txt | cut -d= -f2); OpenMP $(grep -E '^GGML_OPENMP_ENABLED:' $b/CMakeCache.txt | cut -d= -f2); producer of libllama: $(readelf -p .comment $b/bin/libllama.so.1 2>/dev/null | grep -oE '(GCC|clang)[^]]*' | sort -u | tr '\n' ';' | cut -c1-160); libllama ymm $(objdump -d $b/bin/libllama.so.1 | grep -c '%ymm') zmm $(objdump -d $b/bin/libllama.so.1 | grep -c '%zmm')"
  return $rc
}
EXTRA="-DGGML_OPENMP=OFF" one gccnoomp || exit 1
# hip-config-amd.cmake asks HIP_CXX_COMPILER (default: the host C++ compiler) for its compiler-rt builtins and links
# them into everything that uses hip::device; Ubuntu's clang 19 has none installed, so that question goes to ROCm's clang
EXTRA="-DGGML_OPENMP=OFF -DCMAKE_C_COMPILER=/usr/bin/clang-19 -DCMAKE_CXX_COMPILER=/usr/bin/clang++-19 -DHIP_CXX_COMPILER=$RC/lib/llvm/bin/clang++" one c19 || exit 1
EXTRA="-DGGML_OPENMP=OFF -DCMAKE_C_COMPILER=$RC/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=$RC/lib/llvm/bin/clang++" one c24 || exit 1
log "=== HOSTCC BUILDS DONE ==="
