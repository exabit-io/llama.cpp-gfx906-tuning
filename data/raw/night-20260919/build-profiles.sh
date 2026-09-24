#!/bin/bash
# build-profiles.sh — Stage C: build gfx906-multi and gfx906-single on the v0.4.1 base.
#
# Dedicated worktrees, so the main clone is never checked out from under a running job and the
# survey can build any branch without churn. Sequential at -j 24: two concurrent builds would
# oversubscribe and, more importantly, the host must stay predictable (the GPUs are idle now, but
# the clamp rule means host load and loaded dies never overlap).
set -u
W=/root/night-20260919; R=/root/exabit-llama.cpp
log() { echo "$(date -Is) [build] $*" >> $W/build-profiles.log; }
: > $W/build-profiles.log
rc_all=0
for br in gfx906-multi gfx906-single; do
  wt=/root/wt-$br
  bd=/root/build-$br
  if [ ! -d "$wt/.git" ] && [ ! -f "$wt/.git" ]; then
    git -C $R worktree add "$wt" "$br" >> $W/build-profiles.log 2>&1 || { log "worktree add FAILED for $br"; rc_all=1; continue; }
    log "worktree created: $wt -> $br"
  else
    git -C "$wt" checkout -q "$br" 2>>$W/build-profiles.log || { log "checkout FAILED in $wt"; rc_all=1; continue; }
    log "worktree reused: $wt -> $br"
  fi
  head=$(git -C "$wt" rev-parse --short HEAD)
  ahead=$(git -C "$wt" rev-list --count gfx906-substrate-v041..HEAD)
  log "$br at $head, $ahead commit(s) over substrate — configuring"
  rm -rf "$bd" && mkdir -p "$bd"
  {
    cmake -S "$wt" -B "$bd" -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_HIP_ARCHITECTURES=gfx906 \
      -DGGML_HIP_RCCL=ON `# R3.11: required; upstream default is OFF` \
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
    cmake --build "$bd" -j 24
  } > $W/build-$br.log 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -x "$bd/bin/llama-server" ] && [ -x "$bd/bin/llama-batched-bench" ]; then
    log "$br BUILD OK ($head) — binaries present"
  else
    rc_all=1
    log "$br BUILD FAILED exit=$rc — first errors:"
    grep -m8 -E 'error:' $W/build-$br.log >> $W/build-profiles.log 2>/dev/null
    # Trap T9: a v0.4.1 rebase can merge cleanly and still not compile. The log is the only truth.
  fi
done
log "=== BUILD-PROFILES DONE rc=$rc_all ==="
echo "=== BUILD-PROFILES DONE ===" >> $W/build-profiles.log
exit $rc_all
