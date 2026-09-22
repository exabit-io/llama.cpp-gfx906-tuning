#!/bin/bash
# After the fan-power measurement (which needs an idle box), verify the v0.4.1 rebase COMPILES.
# This is the first real gate on the 11 conflict resolutions in the squash-merge.
set -u
W=/root/night-20260919; R=/root/exabit-llama.cpp
while pgrep -f "^/bin/bash \./fanpower[.]sh|^/bin/bash $W/fanpower[.]sh" >/dev/null 2>&1; do sleep 15; done
echo "$(date -Is) fanpower finished; starting build of gfx906-substrate-v041" >> $W/chain.log
cd $R || exit 1
git checkout -q gfx906-substrate-v041 || { echo "$(date -Is) checkout failed" >> $W/chain.log; exit 1; }
rm -rf /root/build-substrate-v041 && mkdir -p /root/build-substrate-v041
{
  cmake -S $R -B /root/build-substrate-v041 -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_HIP_ARCHITECTURES=gfx906 \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
  cmake --build /root/build-substrate-v041 -j 24
} > $W/build-substrate-v041.log 2>&1
rc=$?
echo "$(date -Is) build gfx906-substrate-v041 exit=$rc" >> $W/chain.log
if [ $rc -eq 0 ]; then
  echo "$(date -Is) BUILD OK — the 11 conflict resolutions compile" >> $W/chain.log
else
  echo "$(date -Is) BUILD FAILED — first errors:" >> $W/chain.log
  grep -m8 -E 'error:|Error' $W/build-substrate-v041.log >> $W/chain.log
fi
echo "$(date -Is) === BUILD PHASE DONE ===" >> $W/chain.log
