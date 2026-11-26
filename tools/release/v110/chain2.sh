#!/bin/bash
# chain2.sh -- after chain1: build the final merge branch (merge-v1.1.0 with its three fix commits, snapshots default)
# with ROCm 10.0's compiler and run its final verification (queue-final.sh). Nothing else runs on the host meanwhile.
set -u
W=/root/rebase-v110; P=$W/chain2.progress
log(){ echo "$(date -Is) [chain2] $*" | tee -a $P; }
until grep -q 'CHAIN DONE' $W/chain1.progress 2>/dev/null; do sleep 20; done
log "chain1 is done; building $(git -C /root/wt-merge-v110 rev-parse --short HEAD)"
[ -z "$(git -C /root/wt-merge-v110 status --porcelain --untracked-files=no)" ] || { log "FATAL: /root/wt-merge-v110 is dirty"; exit 1; }
(UIVER=b12774 bash $W/build-rocm.sh /root/wt-merge-v110 /root/build-merge-v110 $W/build-merge-v110.log /opt/rocm/core-10.0 > $W/build-merge-v110.out 2>&1; echo "rc=$?" >> $W/build-merge-v110.out)
log "build: $(tail -1 $W/build-merge-v110.out), errors $(grep -c ' error: ' $W/build-merge-v110.log)"
grep -q '^rc=0' $W/build-merge-v110.out || exit 1
bash $W/queue-final.sh /root/build-merge-v110 merge /root/wt-merge-v110 > $W/queue-final-merge.out 2>&1; log "queue-final merge rc=$?"
log "=== CHAIN2 DONE ==="
