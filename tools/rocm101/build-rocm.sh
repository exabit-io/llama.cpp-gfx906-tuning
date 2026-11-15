#!/bin/bash
# build-rocm.sh SRC BUILD LOG ROCMROOT -- gfx906 HIP build with tests and the tree's own defaults (as tools/release/build.sh),
# against ONE named ROCm tree (/opt/rocm/core-10.1, /opt/rocm/core-10.0): with two streams installed side by side,
# /opt/rocm follows the alternatives, so every ROCm path is given explicitly and the result is checked.
# Optional, from the environment (host code only: the HIP sources take CMAKE_HIP_FLAGS, which nothing here sets):
#   HOSTFLAGS  CMAKE_C_FLAGS and CMAKE_CXX_FLAGS, e.g. '-march=native' (the deb build's host flags)
#   RELFLAGS   CMAKE_C_FLAGS_RELEASE and CMAKE_CXX_FLAGS_RELEASE (default of CMake: '-O3 -DNDEBUG')
#   SHLINK     CMAKE_SHARED_LINKER_FLAGS
#   EXTRA      further cmake arguments, words without spaces
set -u
SRC=$1; B=$2; LOG=$3; ROCM=$4
[ -x $ROCM/lib/llvm/bin/clang++ ] || { echo "no HIP compiler under $ROCM" >&2; exit 2; }
X=()
[ -n "${HOSTFLAGS:-}" ] && X+=("-DCMAKE_C_FLAGS=$HOSTFLAGS" "-DCMAKE_CXX_FLAGS=$HOSTFLAGS")
[ -n "${RELFLAGS:-}" ] && X+=("-DCMAKE_C_FLAGS_RELEASE=$RELFLAGS" "-DCMAKE_CXX_FLAGS_RELEASE=$RELFLAGS")
[ -n "${SHLINK:-}" ] && X+=("-DCMAKE_SHARED_LINKER_FLAGS=$SHLINK")
start=$(date +%s)
{
  ROCM_PATH=$ROCM HIP_PATH=$ROCM cmake -S "$SRC" -B "$B" -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_HIP_ARCHITECTURES=gfx906 \
    -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$ROCM -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache "${X[@]}" ${EXTRA:-} &&
  ROCM_PATH=$ROCM HIP_PATH=$ROCM nice -n 5 cmake --build "$B" -j 56
} > "$LOG" 2>&1
rc=$?
echo "BUILD rc=$rc elapsed=$(( $(date +%s) - start ))s warnings=$(grep -c ' warning: ' "$LOG") errors=$(grep -c ' error: ' "$LOG")" | tee -a "$LOG"
[ $rc = 0 ] || exit $rc
# what the build really used
echo "packages: $(grep -E '^(hip|hipblas|rocblas|rccl)_DIR:' "$B/CMakeCache.txt" | cut -d= -f2 | tr '\n' ' ')" | tee -a "$LOG"
other=$(grep -E '^(hip|hipblas|rocblas|rccl)_DIR:' "$B/CMakeCache.txt" | grep -vc "=$ROCM/")
echo "package dirs outside $ROCM: $other; -isystem/-I ROCm paths in compile commands outside $ROCM: $(grep -oE '(-isystem |-I)/opt/rocm[^ "]*' "$B/compile_commands.json" 2>/dev/null | grep -vc "$ROCM")" | tee -a "$LOG"
exit $other
