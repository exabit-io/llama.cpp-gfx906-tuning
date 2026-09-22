#!/bin/bash
# Build + gate the c4-series (base + 20 of our terms). Chained by SEQUENCE, never by pattern (T4b).
set -u; W=/root/night-20260919; R=/root/exabit-llama.cpp
# wait for the gates by their marker, not by pgrep
while ! grep -q '=== GATES DONE ===' $W/chain.log 2>/dev/null; do sleep 30; done
echo "$(date -Is) gates done; building c4-series" >> $W/chain.log
cd $R && git checkout -q c4-series || exit 1
rm -rf /root/build-c4series && mkdir -p /root/build-c4series
{ cmake -S $R -B /root/build-c4series -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_HIP_ARCHITECTURES=gfx906 \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
  cmake --build /root/build-c4series -j 24; } > $W/build-c4series.log 2>&1
rc=$?
echo "$(date -Is) c4-series build exit=$rc" >> $W/chain.log
[ $rc -ne 0 ] && grep -m6 -E 'error:' $W/build-c4series.log >> $W/chain.log
echo "$(date -Is) === C4 BUILD DONE ===" >> $W/chain.log
