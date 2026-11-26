#!/bin/bash
# queue-final.sh BUILD TAG SRC -- final verification of a tree before it is pushed (ROCm 10.0 compiler and runtime):
# the merge suite with test-backend-ops on four dies, [fold check when FOLD=1,] the fixture tests with the default
# rollback and again with LLAMA_RS_REPLAY=1, the MTP row paths, the carried-row check, the rollback modes.
# Expected (the numbers of the same tree before its last two changes, 2026-11-26): test-backend-ops 21871/21871 x4;
# test-llama-context 148/150 with the two KV-replica models failing cp_h/cp_d(/frag) in BOTH rollback modes;
# mtp-carry: the same accepted/drafted counts for every request A; rs-modes3: its header's four rules.
set -u
B=$1; T=$2; SRC=$3
W=/root/rebase-v110; export RT=/opt/rocm/core-10.0
P=$W/queue-final-$T.progress
log(){ echo "$(date -Is) [queue-final-$T] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is running"; exit 1; }
log "start: $B ($(git -C $SRC rev-parse --short HEAD) $(git -C $SRC status --porcelain --untracked-files=no | wc -l) uncommitted)"
SRC=$SRC bash $W/pr-test.sh $B $W/final-$T-pr-test > $W/final-$T-pr-test.out 2>&1; log "pr-test rc=$?"
log "  allreduce path: $(grep -hiE 'NCCL init failed|probe AllReduce failed' $W/final-$T-pr-test/*.log 2>/dev/null | sort | uniq -c | tr '\n' ' ' | cut -c1-200) (empty = no failure line)"
if [ "${FOLD:-0}" = 1 ]; then bash $W/fold-check.sh $B $W/final-$T-fold > $W/final-$T-fold.out 2>&1; log "fold-check rc=$?"; fi
bash $W/ctx-check.sh $B $W/final-$T-ctx > $W/final-$T-ctx.out 2>&1; log "ctx-check (default rollback) rc=$?"
LLAMA_RS_REPLAY=1 bash $W/ctx-check.sh $B $W/final-$T-ctx-replay > $W/final-$T-ctx-replay.out 2>&1; log "ctx-check (LLAMA_RS_REPLAY=1) rc=$?"
bash $W/mtp-modes.sh $B $W/final-$T-mtp-modes > $W/final-$T-mtp-modes.out 2>&1; log "mtp-modes rc=$?"
bash $W/mtp-carry.sh $B $W/final-$T-mtp-carry > $W/final-$T-mtp-carry.out 2>&1; log "mtp-carry rc=$?"
bash $W/rs-modes3.sh $B $W/final-$T-rs-modes3 > $W/final-$T-rs-modes3.out 2>&1; log "rs-modes3 rc=$?"
log "=== QUEUE DONE ==="
