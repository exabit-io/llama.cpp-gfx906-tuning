#!/bin/bash
# deltaminus.sh — build substrate-MINUS-one-unit, the workhorse for the 65-unit screen.
#
#   deltaminus.sh commit <sha>              a separable commit (53 of 138 reverse-apply cleanly)
#   deltaminus.sh feature <name> <path>...  an entangled group: revert these paths to v0.4.1 content
#
# Why two modes: only 53 of the fork's 138 code commits reverse-apply against the squashed substrate
# as discrete patches (measured 2026-09-20). The other 85 were rewritten by later commits, so they are
# only separable at the granularity of the files they own -- 12 feature units cover them.
#
# NEVER run this while a measurement is in flight. Not for power (setmax caps the host at 150 W) but
# because host load perturbs benchmark timing, and every gate in this campaign was measured on an idle
# host. Comparability is the whole point of a common base.
set -u
W=/root/night-20260919; R=/root/exabit-llama.cpp; WT=/root/wt-sep-test
BASE=gfx906-substrate-v041
mode=${1:?usage: deltaminus.sh commit <sha> | feature <name> <path>...}
log(){ echo "$(date -Is) [dm] $*" | tee -a $W/deltaminus.log; }
busy=$(pgrep -cf '^/root/build-[^ ]*/bin/llama-batched-bench' || true)
if [ "${busy:-0}" -gt 0 ]; then log "REFUSING: a measurement is running; host load would perturb it"; exit 3; fi
git -C $WT checkout -qf --detach $BASE && git -C $WT clean -qfd
case "$mode" in
  commit)
    sha=${2:?need a sha}; tag="minus-$(echo $sha | cut -c1-9)"
    subj=$(git -C $R log -1 --format=%s $sha)
    log "$tag: reverse-applying $sha — $subj"
    if ! git -C $R show --format= --binary "$sha" | git -C $WT apply -R --index --whitespace=nowarn -; then
      log "$tag: NOT SEPARABLE — reverse-apply failed. Bin as entangled; use feature mode."; exit 4
    fi
    ;;
  feature)
    tag="minus-${2:?need a name}"; shift 2
    log "$tag: reverting $# path(s) to v0.4.1 content"
    for p in "$@"; do
      if git -C $R cat-file -e "v0.4.1:$p" 2>/dev/null; then
        git -C $R show "v0.4.1:$p" > "$WT/$p" || { log "$tag: failed to restore $p"; exit 4; }
      else
        rm -f "$WT/$p" || true          # the fork ADDED this file: removing it is the revert
        log "  $p did not exist at v0.4.1 — removed"
      fi
    done
    ;;
  *) log "unknown mode '$mode'"; exit 2;;
esac
bd=/root/build-$tag
rm -rf "$bd" && mkdir -p "$bd"
log "$tag: configuring and building (ccache)"
{
  cmake -S "$WT" -B "$bd" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_HIP_ARCHITECTURES=gfx906 \
    -DGGML_HIP_RCCL=ON `# R3.11: required; upstream default is OFF` \
    -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON \
    "-DGGML_CUDA_FA_QUANTS=all" `# MUST match the comparison arm build-faq:
       # otherwise a delta-minus unit differs from the bundle in TWO ways -- the removed commit AND the
       # compiled FA kernel set -- which breaks the one-variable discipline the whole algebra rests on` \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
  cmake --build "$bd" -j 24
} > $W/build-$tag.log 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -x "$bd/bin/llama-batched-bench" ]; then
  if ! /root/llama.cpp-benchmarking/tools/assert-build-config.sh "$bd" >> $W/deltaminus.log 2>&1; then
    log "$tag: BUILD VIOLATES R3.11 (no RCCL linked) — refusing to hand it to a measurement"
    exit 5
  fi
  log "$tag: BUILD OK -> $bd"
  # Trap T9: a clean revert can still fail to compile. That is itself a verdict:
  # the unit is load-bearing for the rest of the substrate -> bin neutral-required-substrate.
else
  log "$tag: BUILD FAILED exit=$rc — first errors:"; grep -m6 -E 'error:' $W/build-$tag.log >> $W/deltaminus.log
  log "$tag: a unit that cannot be removed without breaking the build is REQUIRED SUBSTRATE, not a loser"
fi
exit $rc
