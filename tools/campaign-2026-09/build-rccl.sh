#!/bin/bash
# build-rccl.sh — rebuild the survey's two builds with GGML_HIP_RCCL=ON.
#
# WHY (lead, 2026-09-21): this is a multi-node cluster programme — Mellanox ConnectX IB cards go into
# PCIe bays 5/6/7 once the patchset work lands, for large-model multi-node testing. RCCL is the
# collective library that plan depends on, and every build in this campaign has had it OFF.
#
# Nobody disabled it: `option(GGML_HIP_RCCL ... OFF)` is upstream's default and I inherited it without
# auditing the build config against the stated requirement. Consequence for the survey: with RCCL
# absent, the "AllReduce off" arm fell back to the META-BACKEND BUTTERFLY path, not to RCCL. Every
# custom-AR verdict so far is therefore measured against the wrong baseline for a production build
# that will ship with RCCL.
#
# What this does NOT fix, stated so it is not assumed: RCCL ON does not give multi-node tensor
# parallelism. The TP communicator is still built from cudaGetDeviceCount() inside one process, with
# no ncclGetUniqueId broadcast and no rank/world bootstrap anywhere in ggml. RCCL will drive the four
# LOCAL dies. Spanning nodes needs cross-process communicator setup added to ggml — a project, not a
# flag — and that is worth scoping separately before the cards arrive.
set -u
W=/root/night-20260919; R=/root/exabit-llama.cpp
log(){ echo "$(date -Is) [rccl] $*" | tee -a $W/build-rccl.log; }
: > $W/build-rccl.log
busy=$(pgrep -cf '^/root/build-[^ ]*/bin/llama-' || true)
[ "${busy:-0}" -gt 0 ] && { log "REFUSING: a measurement is running"; exit 3; }
rc_all=0
for spec in "substrate-v041:gfx906-substrate-v041:/root/wt-sep-test" "c4series:c4-series:/root/exabit-llama.cpp"; do
  name=${spec%%:*}; rest=${spec#*:}; br=${rest%%:*}; wt=${rest##*:}
  bd=/root/build-$name-rccl
  log "$name: checking out $br in $wt"
  git -C "$wt" checkout -qf "$br" 2>>$W/build-rccl.log || { log "$name: checkout FAILED"; rc_all=1; continue; }
  rm -rf "$bd" && mkdir -p "$bd"
  {
    cmake -S "$wt" -B "$bd" -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_HIP_ARCHITECTURES=gfx906 \
      -DGGML_HIP_RCCL=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
    cmake --build "$bd" -j 24
  } > $W/build-$name-rccl.log 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -x "$bd/bin/llama-batched-bench" ]; then
    n=$(strings "$bd"/bin/libggml-hip.so* 2>/dev/null | grep -cE 'ncclCommInit|ncclAllReduce')
    log "$name: BUILD OK -> $bd  (RCCL symbols in libggml-hip: $n)"
    [ "${n:-0}" -eq 0 ] && { log "$name: WARNING — built with GGML_HIP_RCCL=ON but no RCCL symbols linked"; rc_all=1; }
  else
    rc_all=1; log "$name: BUILD FAILED exit=$rc — first errors:"
    grep -m6 -E 'error:' $W/build-$name-rccl.log >> $W/build-rccl.log 2>/dev/null
  fi
done
log "=== BUILD-RCCL DONE rc=$rc_all ==="
touch $W/.build-rccl-done
exit $rc_all
