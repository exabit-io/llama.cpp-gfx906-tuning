#!/bin/bash
# suite-101.sh -- correctness suite of mx-llama.cpp master (df2c6e5c5) built against ROCm 10.1 (/root/build-master-v101),
# the scripts of the v1.0.0 cycle unchanged: pr-test (test-backend-ops on all four dies, PPL 16K x 6, -sm layer -lm dio,
# MTP, Flash-Next), the fold check, the fixture tests. Reference: the same commit on ROCm 10.0, same day
# (reports/2026-11-14-v100-rebase.md). Speeds in this run are NOT comparable: a PyTorch build shares the host.
set -u
W=/root/rocm101-migration-20261114; S=/root/rebase-v100; B=/root/build-master-v101
P=$W/suite-101.progress
log(){ echo "$(date -Is) [suite] $*" | tee -a $P; }
log "start: $B, $(ldd $B/bin/libggml-hip.so | grep libamdhip64 | awk '{print $3}' | xargs readlink -f)"
SRC=/root/wt-master-v100 bash $S/pr-test.sh $B $W/pr-test-101 > $W/pr-test-101.out 2>&1; log "pr-test rc=$?"
log "  allreduce path: $(grep -hiE 'NCCL init failed|probe AllReduce failed' $W/pr-test-101/*.log 2>/dev/null | sort | uniq -c | tr '\n' ' ' | cut -c1-200) (empty = RCCL up)"
bash $S/fold-check.sh $B $W/fold-101 > $W/fold-101.out 2>&1; log "fold-check rc=$?"
bash $S/ctx-check.sh $B $W/ctx-101 > $W/ctx-101.out 2>&1; log "ctx-check rc=$?"
log "=== SUITE DONE ==="
