#!/bin/bash
# gate9-build.sh -- the one build gate9.sh needs beside the existing ones: master-v110 with MMVQ_MAX_BATCH_SIZE 8 (the
# limit gate8 chose) built WITHOUT RCCL (-DGGML_HIP_RCCL=OFF), everything else as the substrate comparison's arms (ROCm
# 10.0 compiler, the tree's defaults, web UI b12774), from /root/wt-bisect: the one line is changed there, the build
# runs, the line is restored. Refuses beside a GPU job.
set -u
W=/root/rebase-v110; G=/root/exabit-llama.cpp; WT=/root/wt-bisect; ROCM=/opt/rocm/core-10.0; P=$W/gate9-build.progress
B=/root/build-cand-norccl; H=ggml/src/ggml-cuda/mmvq.cuh
log(){ echo "$(date -Is) [gate9-build] $*" | tee -a $P; }
for p in llama-server llama-batched-b llama-perplexit test-backend-op test-llama-cont llama-cli cmake; do
  pgrep -x $p >/dev/null && { log "FATAL: $p is running (no builds beside GPU jobs, one build at a time)"; exit 1; }
done
busy=$(cat /sys/class/drm/card[0-9]/device/gpu_busy_percent 2>/dev/null | sort -n | tail -1)
[ "${busy:-0}" -lt 20 ] || { log "FATAL: a die is $busy % busy"; exit 1; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { log "FATAL: $WT has uncommitted changes"; exit 1; }
c=$(git -C $G rev-parse master-v110)
git -C $WT checkout -q --detach $c || { log "FATAL: checkout of master-v110"; exit 1; }
grep -q '^#define MMVQ_MAX_BATCH_SIZE 16 ' $WT/$H || { log "FATAL: $H does not define MMVQ_MAX_BATCH_SIZE 16"; exit 1; }
sed -i 's/^#define MMVQ_MAX_BATCH_SIZE 16 /#define MMVQ_MAX_BATCH_SIZE 8 /' $WT/$H
log "building ${c:0:9} with $(grep '^#define MMVQ_MAX_BATCH_SIZE' $WT/$H | cut -c1-32) and RCCL off into $B"
rc=0
if EXTRA="-DGGML_HIP_RCCL=OFF" UIVER=b12774 bash $W/build-rocm.sh $WT $B $W/build-cand-norccl.log $ROCM > $W/build-cand-norccl.out 2>&1; then
  echo $c > $B/.src-commit; echo "MMVQ_MAX_BATCH_SIZE 8 (one line of $H changed), GGML_HIP_RCCL=OFF" > $B/.src-note
  log "built; RCCL OFF in the cache: $(grep -c '^GGML_HIP_RCCL:BOOL=OFF' $B/CMakeCache.txt), max-ilp in the HIP flags: $(grep -c 'amdgpu-sched-strategy=max-ilp' $B/CMakeCache.txt), librccl linked: $(LD_LIBRARY_PATH=$B/bin:$ROCM/lib ldd $B/bin/libggml-hip.so | grep -c librccl)"
else
  log "BUILD FAILED: $(grep -m2 ' error: \|CMake Error' $W/build-cand-norccl.log | tr '\n' ' ' | cut -c1-300)"; rc=1
fi
git -C $WT checkout -q -- $H; [ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || log "note: $WT not clean after the restore"
log "=== GATE9-BUILD DONE rc=$rc ==="
exit $rc
