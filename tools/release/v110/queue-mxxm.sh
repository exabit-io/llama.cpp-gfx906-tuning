#!/bin/bash
# queue-mxxm.sh -- mxxm-t's master as he left it (557a6068f = his own merge of b12500 + four commits), built by ROCm
# 10.0's compiler and run on the ROCm 10.0 runtime, i.e. the platform of PR #24's numbers: the merge suite, the fixture
# tests, the four MTP row paths. Reference: our merge-v1.0.0 build on the same platform (reports/2026-11-14-v100-rebase.md):
# test-backend-ops 20388/20388 x4 (his tree has more cases: upstream added tests up to b12500), PPL 16K/6 5.6126,
# MTP 154/200, test-llama-context 142 or 143 of 144 on four dies (KV replicas), identical text on the four MTP paths.
# Speeds are smoke values.
set -u
W=/root/rebase-v110; B=/root/build-mxxm-b12500; export RT=/opt/rocm/core-10.0
P=$W/queue-mxxm.progress
log(){ echo "$(date -Is) [queue-mxxm] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x test-backend-op >/dev/null && { log "FATAL: test-backend-ops is running"; exit 1; }
log "start: $B, hip runtime $(LD_LIBRARY_PATH=$B/bin:$B/lib:$RT/lib ldd $B/bin/libggml-hip.so | awk '/libamdhip64/{print $3}' | xargs readlink -f)"
SRC=/root/wt-mxxm-b12500 bash $W/pr-test.sh $B $W/pr-test-mxxm > $W/pr-test-mxxm.out 2>&1; log "pr-test rc=$?"
bash $W/ctx-check.sh $B $W/ctx-mxxm > $W/ctx-mxxm.out 2>&1; log "ctx-check rc=$?"
bash $W/mtp-modes.sh $B $W/mtp-modes-mxxm > $W/mtp-modes-mxxm.out 2>&1; log "mtp-modes rc=$?"
log "=== QUEUE DONE ==="
