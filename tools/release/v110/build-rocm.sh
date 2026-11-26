#!/bin/bash
# build-rocm.sh SRC BUILD LOG ROCMROOT -- gfx906 HIP build with tests and the tree's own defaults (as tools/release/build.sh),
# against ONE named ROCm tree (/opt/rocm/core-10.1, /opt/rocm/core-10.0): with two streams installed side by side,
# /opt/rocm follows the alternatives, so every ROCm path is given explicitly and the result is checked.
# Optional, from the environment, for the host code:
#   HOSTFLAGS  CMAKE_C_FLAGS and CMAKE_CXX_FLAGS, e.g. '-march=native' (the deb build's host flags)
#   RELFLAGS   CMAKE_C_FLAGS_RELEASE and CMAKE_CXX_FLAGS_RELEASE (default of CMake: '-O3 -DNDEBUG')
#   SHLINK     CMAKE_SHARED_LINKER_FLAGS
#   EXTRA      further cmake arguments, words without spaces
# Optional, for the HIP (GPU) sources:
#   HIPFLAGS   CMAKE_HIP_FLAGS. The tree appends its own '-mllvm -amdgpu-sched-strategy=max-ilp' to it, but only while the
#              architecture is exactly 'gfx906' (GGML_HIP_GFX906_MAX_ILP); with another HIPARCH name it here if wanted
#   HIPREL     CMAKE_HIP_FLAGS_RELEASE (default of CMake: '-O3 -DNDEBUG')
#   UIVER      upstream build tag of the web UI assets to embed, e.g. b12774 (see below)
#   HIPARCH    the HIP architecture (default gfx906); a full target ID such as 'gfx906:sramecc-:xnack-' is passed as is
set -u
SRC=$1; B=$2; LOG=$3; ROCM=$4
[ -x $ROCM/lib/llvm/bin/clang++ ] || { echo "no HIP compiler under $ROCM" >&2; exit 2; }
X=()
[ -n "${HOSTFLAGS:-}" ] && X+=("-DCMAKE_C_FLAGS=$HOSTFLAGS" "-DCMAKE_CXX_FLAGS=$HOSTFLAGS")
[ -n "${RELFLAGS:-}" ] && X+=("-DCMAKE_C_FLAGS_RELEASE=$RELFLAGS" "-DCMAKE_CXX_FLAGS_RELEASE=$RELFLAGS")
[ -n "${SHLINK:-}" ] && X+=("-DCMAKE_SHARED_LINKER_FLAGS=$SHLINK")
# The HIP compiler takes its libstdc++ from the GCC installation the HOST C++ compiler belongs to. clang picks the newest
# GCC directory it finds; since 2026-11-17 this box has gcc-14 without libstdc++-14-dev (installed for a kernel-module
# build), and both ROCm compilers then fail at "cmath not found". Naming the directory keeps the headers every earlier
# build used (GCC 13) and makes the choice independent of what else is installed.
GCCDIR=$(dirname "$(${CXX:-c++} -print-libgcc-file-name)")
SEL=$($ROCM/lib/llvm/bin/clang++ -v -fsyntax-only -x c++ /dev/null 2>&1 | sed -n 's/^Selected GCC installation: //p')
if [ -n "$GCCDIR" ] && [ "$SEL" != "$GCCDIR" ]; then
  HIPFLAGS="--gcc-install-dir=$GCCDIR${HIPFLAGS:+ $HIPFLAGS}"
  echo "HIP compiler would select $SEL; pinned to the host compiler's $GCCDIR" >&2
fi
[ -n "${HIPFLAGS:-}" ] && X+=("-DCMAKE_HIP_FLAGS=$HIPFLAGS")
[ -n "${HIPREL:-}" ] && X+=("-DCMAKE_HIP_FLAGS_RELEASE=$HIPREL")
ARCH=${HIPARCH:-gfx906}
# Since b124xx the server's web UI is not in the tree: the build downloads it from upstream's bucket by BUILD NUMBER
# (b<commit count>). A fork's count names no upstream build, the script then takes "latest", and a "latest" newer than
# the tree fails the asset check ("missing required asset(s)"). UIVER names the upstream build the tree is merged up to.
[ -n "${UIVER:-}" ] && export HF_UI_VERSION=$UIVER
start=$(date +%s)
{
  ROCM_PATH=$ROCM HIP_PATH=$ROCM cmake -S "$SRC" -B "$B" -DGGML_HIP=ON "-DAMDGPU_TARGETS=$ARCH" "-DCMAKE_HIP_ARCHITECTURES=$ARCH" \
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
