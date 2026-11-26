#!/bin/bash
# queue-master.sh -- master-v110 (the v1.1.0 merge with its two fixes + the 22 Exabit commits), ROCm 10.0 compiler and
# runtime (/root/build-master-v110): merge suite with test-backend-ops on four dies, the fold check, the fixture tests,
# the MTP row paths, the rollback modes. Reference, the v1.1.0 merge the same day (queue-merge, queue-merge2):
# test-backend-ops 21871/21871 x4, PPL 16K/6 5.6130, test-llama-context 148/150 (KV-replica models), replay 15/15.
# Expected of master as in every cycle: every test-backend-ops case passes; PPL a little BELOW the merge's and the same
# with the folds on and off except the GDN L2 fold (fold-check.sh says which); text of the MTP row paths identical.
# v1.0.0 values, for the size of the fold effect only: merge 5.6126, master 5.6093; folds on 5.0822 / off 5.0575 (2K x 8).
set -u
W=/root/rebase-v110; B=/root/build-master-v110; export RT=/opt/rocm/core-10.0
P=$W/queue-master.progress
log(){ echo "$(date -Is) [queue-master] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is running"; exit 1; }
log "start: $B ($(git -C /root/wt-master-v110 rev-parse --short HEAD)), hip runtime $(LD_LIBRARY_PATH=$B/bin:$B/lib:$RT/lib ldd $B/bin/libggml-hip.so | awk '/libamdhip64/{print $3}' | xargs readlink -f)"
SRC=/root/wt-master-v110 bash $W/pr-test.sh $B $W/pr-test-master > $W/pr-test-master.out 2>&1; log "pr-test rc=$?"
log "  allreduce path: $(grep -hiE 'NCCL init failed|probe AllReduce failed' $W/pr-test-master/*.log 2>/dev/null | sort | uniq -c | tr '\n' ' ' | cut -c1-200) (empty = RCCL up)"
bash $W/fold-check.sh $B $W/fold-master > $W/fold-master.out 2>&1; log "fold-check rc=$?"
bash $W/ctx-check.sh $B $W/ctx-master > $W/ctx-master.out 2>&1; log "ctx-check rc=$?"
bash $W/mtp-modes.sh $B $W/mtp-modes-master > $W/mtp-modes-master.out 2>&1; log "mtp-modes rc=$?"
bash $W/rs-modes2.sh $B $W/rs-modes2-master > $W/rs-modes2-master.out 2>&1; log "rs-modes2 rc=$?"
log "=== QUEUE DONE ==="
