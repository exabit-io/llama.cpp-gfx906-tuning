#!/bin/bash
# build-stock.sh — the D2 ZERO POINT: stock upstream v0.4.1, no gfx906 commits at all.
# Gate 1 exists because every baseline figure so far is on the substrate build. Without this, the
# algebra T = B + R + sum(Pi) has no fixed B and every delta is measured against a moving reference.
set -u
W=/root/night-20260919; R=/root/exabit-llama.cpp
wt=/root/wt-stock-v041; bd=/root/build-stock-v041
log() { echo "$(date -Is) [stock] $*" >> $W/build-stock.log; }
: > $W/build-stock.log
if [ ! -e "$wt/.git" ]; then
  git -C $R worktree add --detach "$wt" v0.4.1 >> $W/build-stock.log 2>&1 || { log "worktree add FAILED"; exit 1; }
fi
git -C "$wt" checkout -q --detach v0.4.1 2>>$W/build-stock.log
head=$(git -C "$wt" rev-parse --short HEAD)
# Assert this really is stock: zero commits from the substrate branch may be present.
extra=$(git -C $R rev-list --count v0.4.1..HEAD 2>/dev/null || echo 0)
log "worktree at $head (tag v0.4.1); gfx906 commits present: $(git -C "$wt" log --oneline v0.4.1 -1 --format=%h)"
rm -rf "$bd" && mkdir -p "$bd"
{
  cmake -S "$wt" -B "$bd" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_HIP_ARCHITECTURES=gfx906 \
    -DGGML_HIP_RCCL=ON `# R3.11: required; upstream default is OFF` \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
  cmake --build "$bd" -j 24
} > $W/build-stock-v041.log 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -x "$bd/bin/llama-batched-bench" ]; then
  log "STOCK BUILD OK ($head)"
else
  log "STOCK BUILD FAILED exit=$rc — first errors:"; grep -m8 -E 'error:' $W/build-stock-v041.log >> $W/build-stock.log 2>/dev/null
fi
log "=== BUILD-STOCK DONE rc=$rc ==="
exit $rc
