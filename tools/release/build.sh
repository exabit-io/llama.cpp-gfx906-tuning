#!/bin/bash
# build.sh SRC BUILD LOG -- gfx906 HIP build with tests, same options as build-mx-both (the tree's own defaults)
set -u
SRC=$1; B=$2; LOG=$3
start=$(date +%s)
{
  cmake -S "$SRC" -B "$B" -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_HIP_ARCHITECTURES=gfx906 \
    -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache &&
  nice -n 5 cmake --build "$B" -j 56
} > "$LOG" 2>&1
rc=$?
echo "BUILD rc=$rc elapsed=$(( $(date +%s) - start ))s warnings=$(grep -c ' warning: ' "$LOG") errors=$(grep -c ' error: ' "$LOG")" | tee -a "$LOG"
exit $rc
