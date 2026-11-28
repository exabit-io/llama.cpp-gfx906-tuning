#!/bin/bash
# attr8-build.sh -- the cut-point builds of attr8.sh: master-v110's 22 Exabit commits cut into groups, one build per cut,
# each as the substrate comparison's arms (ROCm 10.0 compiler, the tree's own defaults plus -DGGML_HIP_RCCL=ON, web UI
# b12774), from /root/wt-bisect checked out at the cut, one after the other (a checkout only after the build before it
# has ended). Builds only what is missing; refuses beside a GPU job (no host load beside loaded dies).
# DRY=1: prints the cuts and checks their order, builds nothing.
#   a0  merge a3aa1c772                          /root/build-sub-merge (exists)
#   a1  567a2a769  + RCCL default, AR size gate and its default, FA_QUANTS=all default (4 commits)
#   a2  0557bdc31  + the folds: terms 02 03 04 07 08 10 12 (7)
#   a3  f6142d563  + DPP warp reductions: terms 23 28 (2)
#   a4  8ff1e7c9c  + MMVQ: terms 01 15 05 09 (4)
#   a5  430835223  + max-ilp scheduler default (1)
#   a6  master 9a7177d0a  + LLVM 24 tile rule x3, no GDN producer fold on replay inputs (4)   /root/build-master-v110 (exists)
set -u
W=/root/rebase-v110; G=/root/exabit-llama.cpp; WT=/root/wt-bisect; ROCM=/opt/rocm/core-10.0; P=$W/attr8-build.progress
CUTS="a1=567a2a769 a2=0557bdc31 a3=f6142d563 a4=8ff1e7c9c a5=430835223"
log(){ echo "$(date -Is) [attr8-build] $*" | tee -a $P; }
prev=merge-v1.1.0
for spec in $CUTS; do
  a=${spec%%=*}; c=${spec#*=}
  git -C $G merge-base --is-ancestor $prev $c && git -C $G merge-base --is-ancestor $c master-v110 || { log "FATAL: $a ($c) is not between $prev and master-v110"; exit 1; }
  [ "${DRY:-0}" = 1 ] && echo "$a $c: $(git -C $G rev-list --count $prev..$c) commits after $prev; tip: $(git -C $G log -1 --format=%s $c | cut -c1-90)"
  prev=$c
done
[ "${DRY:-0}" = 1 ] && { echo "a6 master-v110: $(git -C $G rev-list --count $prev..master-v110) commits after $prev"; exit 0; }
for p in llama-server llama-batched-b llama-perplexit test-backend-op test-llama-cont llama-cli; do
  pgrep -x $p >/dev/null && { log "FATAL: $p is running (no builds beside GPU jobs)"; exit 1; }
done
busy=$(cat /sys/class/drm/card[0-9]/device/gpu_busy_percent 2>/dev/null | sort -n | tail -1)
[ "${busy:-0}" -lt 20 ] || { log "FATAL: a die is $busy % busy"; exit 1; }
[ -d $WT ] || { log "FATAL: no worktree $WT"; exit 1; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { log "FATAL: $WT has uncommitted changes"; exit 1; }
rc=0
for spec in $CUTS; do
  a=${spec%%=*}; c=$(git -C $G rev-parse ${spec#*=}); b=/root/build-attr-$a
  if [ -x $b/bin/llama-batched-bench ] && grep -q '^GGML_HIP_RCCL:BOOL=ON' $b/CMakeCache.txt 2>/dev/null && [ "$(cat $b/.src-commit 2>/dev/null)" = "$c" ]; then
    log "$a: present ($b, ${c:0:9})"; continue
  fi
  git -C $WT checkout -q --detach $c || { log "FATAL: checkout of $a (${c:0:9})"; rc=1; break; }
  log "$a: building ${c:0:9} into $b"
  if EXTRA="-DGGML_HIP_RCCL=ON" UIVER=b12774 bash $W/build-rocm.sh $WT $b $W/build-attr-$a.log $ROCM > $W/build-attr-$a.out 2>&1; then
    echo $c > $b/.src-commit
    log "$a: built; RCCL $(grep -c '^GGML_HIP_RCCL:BOOL=ON' $b/CMakeCache.txt), max-ilp in the HIP flags: $(grep -c 'amdgpu-sched-strategy=max-ilp' $b/CMakeCache.txt), librccl linked: $(LD_LIBRARY_PATH=$b/bin:$ROCM/lib ldd $b/bin/libggml-hip.so | grep -c librccl)"
  else
    log "$a: BUILD FAILED (the session runs without this arm): $(grep -m2 ' error: \|CMake Error' $W/build-attr-$a.log | tr '\n' ' ' | cut -c1-300)"; rc=1
  fi
done
git -C $WT checkout -q --detach master-v110 || log "note: $WT not back at master-v110"
log "=== ATTR8-BUILD DONE rc=$rc ==="
exit $rc
