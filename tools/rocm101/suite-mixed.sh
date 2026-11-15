#!/bin/bash
# suite-mixed.sh -- the configuration a node is in when ROCm 10.1 is installed beside 10.0 and a ROCm 10.0 build of
# mx-llama.cpp is still in place: code compiled by ROCm 10.0's toolchain (/root/build-master-v100r, master df2c6e5c5),
# running on ROCm 10.1's libraries (what /opt/rocm/lib selects once 10.1 is installed). rocm-cross.sh measured its speed
# (10.0's numbers); this is its correctness: test-backend-ops on all four dies, PPL 16K x 6, -sm layer -lm dio, MTP,
# Flash-Next, the fold check, the fixture tests.
set -u
W=/root/rocm101-migration-20261114; S=/root/rebase-v100; B=/root/build-master-v100r
P=$W/suite-mixed.progress
log(){ echo "$(date -Is) [mixed] $*" | tee -a $P; }
hip=$(LD_LIBRARY_PATH=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib ldd $B/bin/libggml-hip.so | awk '/libamdhip64/ {print $3}' | xargs readlink -f)
case $hip in /opt/rocm/core-10.1/*) ;; *) log "FATAL: the build would load $hip, not ROCm 10.1"; exit 1;; esac
log "start: $B (compiled by $(grep -m1 -oE 'CMAKE_HIP_COMPILER:[A-Z]*=.*' $B/CMakeCache.txt | cut -d= -f2)) on $hip"
SRC=/root/wt-master-v100 bash $S/pr-test.sh $B $W/pr-test-mixed > $W/pr-test-mixed.out 2>&1; log "pr-test rc=$?"
log "  allreduce path: $(grep -hiE 'NCCL init failed|probe AllReduce failed' $W/pr-test-mixed/*.log 2>/dev/null | sort | uniq -c | tr '\n' ' ' | cut -c1-200) (empty = RCCL up)"
bash $S/fold-check.sh $B $W/fold-mixed > $W/fold-mixed.out 2>&1; log "fold-check rc=$?"
bash $S/ctx-check.sh $B $W/ctx-mixed > $W/ctx-mixed.out 2>&1; log "ctx-check rc=$?"
ARMS="combined" bash $S/mtp-modes.sh $B $W/mtp-mixed > $W/mtp-mixed.out 2>&1; log "MTP np1 rc=$?: $(grep 'run [123]\|long' $W/mtp-mixed/progress | sed 's/.*\] //' | cut -c1-110 | tr '\n' ';')"
log "=== MIXED SUITE DONE ==="
