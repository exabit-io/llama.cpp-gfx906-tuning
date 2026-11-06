#!/bin/bash
# queue4.sh -- master and candidates rebuilt on the PR tip with cdc81dae4 (term 04: folded calls skip the column-thread
# GDN kernel): master suite + test-backend-ops x2 + folds, candidates suite + test-backend-ops x2, MTP chunked-graph A/B.
set -u
W=/root/rebase-v090
MA=/root/build-master-v090; CA=/root/build-candidates-v090
Q=$W/queue4.progress
log(){ echo "$(date -Is) [queue4] $*" | tee -a $Q; }
log "start"
SRC=/root/wt-master-v090 SKIP_TBO=1 bash $W/pr-test.sh $MA $W/pr-test-master3 > $W/pr-test-master3.out 2>&1; log "pr-test master rc=$?"
log "  allreduce path: $(grep -hiE 'NCCL init failed|probe AllReduce failed' $W/pr-test-master3/*.log | sort | uniq -c | tr '\n' ' ' | cut -c1-200) (empty = RCCL up)"
. /root/rocm-tests/bench/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
tbo2(){ # tbo2 BUILD OUT DEV DEV
  local B=$1 O=$2; shift 2; mkdir -p $O; setmax
  for i in "$@"; do LD_LIBRARY_PATH=$B/bin:/opt/rocm/lib timeout 5400 $B/bin/test-backend-ops -b ROCm$i > $O/tbo-$i.log 2>&1 & done
  wait; restore
  for i in "$@"; do log "$(basename $O) ROCm$i: $(grep -E '[0-9]+/[0-9]+ tests passed' $O/tbo-$i.log | tail -1); failed: $(grep -F 'FAIL' $O/tbo-$i.log | grep '(' | sed 's/.*  //' | cut -c1-160 | head -3 | tr '\n' ' ')"; done
}
tbo2 $MA $W/tbo-master3 0 2
bash $W/fold-check.sh $MA $W/fold-master3 > $W/fold-master3.out 2>&1; log "fold-check rc=$?"
bash $W/fold-isolate.sh $MA $W/isolate-master3 > $W/isolate-master3.out 2>&1; log "fold-isolate rc=$?"
SRC=/root/wt-candidates-v090 SKIP_TBO=1 bash $W/pr-test.sh $CA $W/pr-test-candidates3 > $W/pr-test-candidates3.out 2>&1; log "pr-test candidates rc=$?"
tbo2 $CA $W/tbo-candidates3 1 3
bash $W/mtp-ab.sh $MA $W/mtp-ab-master > $W/mtp-ab-master.out 2>&1; log "mtp-ab rc=$?"
log "=== QUEUE4 DONE ==="
