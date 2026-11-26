#!/bin/bash
# queue-merge2.sh -- after the two fix commits on merge-v1.1.0 (63181a17e top-k MoE softmax, e1711b3d6 576/512 tile
# selection): the whole test-backend-ops on four dies and the whole fixture run again, then the numerics attribution
# (numerics.sh: stock b12500, stock v1.1.0, mxxm-t's master, the merge), then the rollback modes with the kernel held
# fixed (rs-modes2.sh). Expected: test-backend-ops N/N on every die; test-llama-context no abort, failures only in the
# models of the known KV-replica defect (cp_h / cp_d / frag of gemma4-dense or mistral-large4-moe).
set -u
W=/root/rebase-v110; B=/root/build-merge-v110; export RT=/opt/rocm/core-10.0
P=$W/queue-merge2.progress
log(){ echo "$(date -Is) [queue-merge2] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is running"; exit 1; }
mkdir -p $W/tbo-merge2
PIDS=""
for i in 0 1 2 3; do
  LD_LIBRARY_PATH=$B/bin:$B/lib:$RT/lib timeout 5400 $B/bin/test-backend-ops -b ROCm$i > $W/tbo-merge2/tbo-$i.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p; done
for i in 0 1 2 3; do log "test-backend-ops ROCm$i: $(grep -aE '[0-9]+/[0-9]+ tests passed' $W/tbo-merge2/tbo-$i.log | tail -1 | tr -s ' '); FAIL lines $(grep -a -c FAIL $W/tbo-merge2/tbo-$i.log)"; done
bash $W/ctx-check.sh $B $W/ctx-merge2 > $W/ctx-merge2.out 2>&1; log "ctx-check rc=$?: $(tail -5 $W/ctx-merge2/progress | grep 'test-llama-context' | cut -c40-420)"
bash $W/numerics.sh $W/numerics /root/build-stock-b12500 /root/build-stock-v110 /root/build-mxxm-b12500 /root/build-merge-v110 > $W/numerics.out 2>&1; log "numerics rc=$?"
log "=== QUEUE DONE ==="
