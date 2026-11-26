#!/bin/bash
# queue-merge.sh -- the v1.1.0 merge build (/root/wt-merge-v110, uncommitted merge of v1.1.0 into mxxm-t 557a6068f), built by
# ROCm 10.0's compiler and run on the ROCm 10.0 runtime (the platform of PR #24's numbers and of queue-mxxm.sh, his master
# the same day): merge suite, fixture tests, the four MTP row paths, and the rollback modes (rs-modes.sh: its reading rule
# is in its header). References: his master today (queue-mxxm.progress): test-backend-ops 20565/20633 x4 (68 TOPK_MOE),
# PPL 16K/6 5.6126, MTP 154/200, test-llama-context 144/145. Speeds are smoke values.
set -u
W=/root/rebase-v110; B=${B:-/root/build-merge-v110}; T=${T:-merge}; export RT=/opt/rocm/core-10.0
P=$W/queue-$T.progress
log(){ echo "$(date -Is) [queue-$T] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x test-backend-op >/dev/null && { log "FATAL: test-backend-ops is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is running"; exit 1; }
log "start: $B, hip runtime $(LD_LIBRARY_PATH=$B/bin:$B/lib:$RT/lib ldd $B/bin/libggml-hip.so | awk '/libamdhip64/{print $3}' | xargs readlink -f)"
SRC=${SRC:-/root/wt-merge-v110} bash $W/pr-test.sh $B $W/pr-test-$T > $W/pr-test-$T.out 2>&1; log "pr-test rc=$?"
bash $W/ctx-check.sh $B $W/ctx-$T > $W/ctx-$T.out 2>&1; log "ctx-check rc=$?"
bash $W/mtp-modes.sh $B $W/mtp-modes-$T > $W/mtp-modes-$T.out 2>&1; log "mtp-modes rc=$?"
bash $W/rs-modes.sh $B $W/rs-modes-$T > $W/rs-modes-$T.out 2>&1; log "rs-modes rc=$?"
log "=== QUEUE DONE ==="
