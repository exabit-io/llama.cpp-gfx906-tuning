#!/bin/bash
# chain1.sh -- the GPU queue of 2026-11-26 evening, one job after the other (each script has its own header and rule):
# waits for queue-merge2, then queue-rs (rollback question, quiet host), mtp-carry on the merge build and on the build
# with the carried-row fix, queue-master (suite on master-v110), mtp-depth-v110 (tile threshold at MTP depth, quiet host).
set -u
W=/root/rebase-v110; P=$W/chain1.progress
log(){ echo "$(date -Is) [chain1] $*" | tee -a $P; }
until grep -q 'QUEUE DONE' $W/queue-merge2.progress 2>/dev/null; do sleep 20; done
log "queue-merge2 is done"
bash $W/queue-rs.sh > $W/queue-rs.out 2>&1; log "queue-rs rc=$?"
RT=/opt/rocm/core-10.0 bash $W/mtp-carry.sh /root/build-merge-v110 $W/mtp-carry-merge > $W/mtp-carry-merge.out 2>&1; log "mtp-carry merge rc=$?"
RT=/opt/rocm/core-10.0 bash $W/mtp-carry.sh /root/build-merge-v110-carry $W/mtp-carry-fix > $W/mtp-carry-fix.out 2>&1; log "mtp-carry fix rc=$?"
bash $W/queue-master.sh > $W/queue-master.out 2>&1; log "queue-master rc=$?"
bash $W/mtp-depth-v110.sh > $W/mtp-depth-v110.out 2>&1; log "mtp-depth-v110 rc=$?"
log "=== CHAIN DONE ==="
