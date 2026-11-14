#!/bin/bash
# queue1.sh -- v1.0.0 cycle, GPU jobs one at a time on an idle host:
#   merge build: MTP row paths (one slot, four slots), extra runtime checks
#   master: suite, test-backend-ops on two dies, folds, MTP row paths
#   candidates: suite, test-backend-ops on two dies
set -u
W=/root/rebase-v100
PR=/root/build-pr-v100; MA=/root/build-master-v100; CA=/root/build-candidates-v100
Q=$W/queue1.progress
log(){ echo "$(date -Is) [queue1] $*" | tee -a $Q; }
log "start"
bash $W/mtp-modes.sh $PR $W/mtp-modes-pr > $W/mtp-modes-pr.out 2>&1; log "mtp-modes merge rc=$?"
ARMS="combined host" bash $W/mtp-modes.sh $PR $W/mtp-modes-pr-np4 4 > $W/mtp-modes-pr-np4.out 2>&1; log "mtp-modes merge np4 rc=$?"
bash $W/extra-checks.sh $PR $W/extra-pr > $W/extra-pr.out 2>&1; log "extra-checks merge rc=$?"
SRC=/root/wt-master-v100 SKIP_TBO=1 bash $W/pr-test.sh $MA $W/pr-test-master > $W/pr-test-master.out 2>&1; log "pr-test master rc=$?"
log "  allreduce path: $(grep -hiE 'NCCL init failed|probe AllReduce failed' $W/pr-test-master/*.log | sort | uniq -c | tr '\n' ' ' | cut -c1-200) (empty = RCCL up)"
. /root/rocm-tests/bench/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
tbo2(){ # tbo2 BUILD OUT DEV DEV
  local B=$1 O=$2; shift 2; mkdir -p $O; setmax
  local pids=""
  for i in "$@"; do LD_LIBRARY_PATH=$B/bin:/opt/rocm/lib timeout 5400 $B/bin/test-backend-ops -b ROCm$i > $O/tbo-$i.log 2>&1 & pids="$pids $!"; done
  for p in $pids; do wait $p; done
  restore
  for i in "$@"; do log "$(basename $O) ROCm$i: $(grep -E '[0-9]+/[0-9]+ tests passed' $O/tbo-$i.log | tail -1); failed: $(grep -F 'FAIL' $O/tbo-$i.log | grep '(' | sed 's/.*  //' | cut -c1-160 | head -3 | tr '\n' ' ')"; done
}
tbo2 $MA $W/tbo-master 0 2
bash $W/fold-check.sh $MA $W/fold-master > $W/fold-master.out 2>&1; log "fold-check rc=$?"
bash $W/fold-isolate.sh $MA $W/isolate-master > $W/isolate-master.out 2>&1; log "fold-isolate rc=$?"
bash $W/mtp-modes.sh $MA $W/mtp-modes-master > $W/mtp-modes-master.out 2>&1; log "mtp-modes master rc=$?"
SRC=/root/wt-candidates-v100 SKIP_TBO=1 bash $W/pr-test.sh $CA $W/pr-test-candidates > $W/pr-test-candidates.out 2>&1; log "pr-test candidates rc=$?"
tbo2 $CA $W/tbo-candidates 1 3
log "=== QUEUE1 DONE ==="
