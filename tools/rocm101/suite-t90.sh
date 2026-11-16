#!/bin/bash
# suite-t90.sh -- correctness suite of mx-llama.cpp 290fc642a (df2c6e5c5 + the LLVM 24 tile rule) built by ROCm 10.1
# (/root/build-v101-t90: GPU code byte-identical to the package 1.0.0+git20261115.290fc642a.rocm10.1, which has no
# test-backend-ops),
# the scripts of the v1.0.0 cycle unchanged: pr-test (test-backend-ops on all four dies, PPL 16K x 6, -sm layer -lm dio,
# MTP, Flash-Next), the fold check, the fixture tests. Reference: df2c6e5c5 on ROCm 10.0, 2026-11-14
# (reports/2026-11-14-v100-rebase.md) and the plain ROCm 10.1 build (suite-101.sh).
set -u
W=/root/rocm101-migration-20261114; S=/root/rebase-v100; B=/root/build-v101-t90
P=$W/suite-t90.progress
log(){ echo "$(date -Is) [suite] $*" | tee -a $P; }
log "start: $B, $(ldd $B/bin/libggml-hip.so | grep libamdhip64 | awk '{print $3}' | xargs readlink -f)"
SRC=/root/wt-master-t90 bash $S/pr-test.sh $B $W/pr-test-t90 > $W/pr-test-t90.out 2>&1; log "pr-test rc=$?"
log "  allreduce path: $(grep -hiE 'NCCL init failed|probe AllReduce failed' $W/pr-test-t90/*.log 2>/dev/null | sort | uniq -c | tr '\n' ' ' | cut -c1-200) (empty = RCCL up)"
bash $S/fold-check.sh $B $W/fold-t90 > $W/fold-t90.out 2>&1; log "fold-check rc=$?"
bash $S/ctx-check.sh $B $W/ctx-t90 > $W/ctx-t90.out 2>&1; log "ctx-check rc=$?"
log "=== SUITE DONE ==="
