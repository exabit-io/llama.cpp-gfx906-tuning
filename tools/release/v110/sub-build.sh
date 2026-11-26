#!/bin/bash
# sub-build.sh -- the builds of the substrate comparison (substrate-v110.sh): mxxm-t's master and the v1.1.0 merge
# (WITH_STOCK=1: also upstream llama.cpp v1.1.0 unmodified), each with the tree's own defaults PLUS -DGGML_HIP_RCCL=ON, by the ROCm 10.0
# compiler (as /root/build-master-v110, the fourth arm, which has RCCL on by its own default). RCCL on in every arm so
# that the four dies exchange through the same library in all of them: stock without it falls back to a host-side
# butterfly on four devices ("internal AllReduce init failed (n_devices != 2?)", numerics/build-stock-v110-ppl-tensor.log),
# which is a build option, not a property of the code.
# Builds only what is missing. Refuses to run while a GPU job is up (no host load beside loaded dies).
set -u
W=/root/rebase-v110; ROCM=/opt/rocm/core-10.0; P=$W/sub-build.progress
log(){ echo "$(date -Is) [sub-build] $*" | tee -a $P; }
for p in llama-server llama-batched-b llama-perplexit test-backend-op test-llama-cont llama-cli; do
  pgrep -x $p >/dev/null && { log "FATAL: $p is running (no builds beside GPU jobs)"; exit 1; }
done
busy=$(cat /sys/class/drm/card[0-9]/device/gpu_busy_percent 2>/dev/null | sort -n | tail -1)
[ "${busy:-0}" -lt 20 ] || { log "FATAL: a die is $busy % busy"; exit 1; }
rc=0
build(){ # build NAME SRC UIVER
  local name=$1 src=$2 ui=$3 b=/root/build-sub-$1
  if [ -x $b/bin/llama-server ] && [ -x $b/bin/llama-batched-bench ] && grep -q '^GGML_HIP_RCCL:BOOL=ON' $b/CMakeCache.txt 2>/dev/null \
     && [ "$(cat $b/.src-commit 2>/dev/null)" = "$(git -C $src rev-parse HEAD)" ]; then
    log "$name: present ($b, $(git -C $src rev-parse --short HEAD))"; return 0
  fi
  [ -z "$(git -C $src status --porcelain --untracked-files=no)" ] || { log "FATAL: $src has uncommitted changes"; rc=1; return 1; }
  log "$name: building $src ($(git -C $src rev-parse --short HEAD)) into $b"
  if EXTRA="-DGGML_HIP_RCCL=ON" UIVER=$ui bash $W/build-rocm.sh $src $b $W/build-sub-$name.log $ROCM > $W/build-sub-$name.out 2>&1; then
    git -C $src rev-parse HEAD > $b/.src-commit
    log "$name: $(tail -4 $W/build-sub-$name.out | head -1); RCCL $(grep -c '^GGML_HIP_RCCL:BOOL=ON' $b/CMakeCache.txt), librccl linked: $(LD_LIBRARY_PATH=$b/bin:$ROCM/lib ldd $b/bin/libggml-hip.so | grep -c librccl)"
  else
    log "$name: BUILD FAILED: $(grep -m2 ' error: \|CMake Error' $W/build-sub-$name.log | tr '\n' ' ' | cut -c1-300)"; rc=1
  fi
}
build mxxm  /root/wt-mxxm-b12500 b12500
build merge /root/wt-merge-v110 b12774
[ "${WITH_STOCK:-0}" = 1 ] && build stock /root/wt-stock-v110 b12774
log "=== SUB-BUILD DONE rc=$rc ==="
exit $rc
