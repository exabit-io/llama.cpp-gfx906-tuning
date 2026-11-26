#!/bin/bash
# queue-rs.sh -- the rollback question, second session (quiet host: its speeds are compared between arms).
#  1. rs-modes.sh on mxxm-t's master (557a6068f), arms ring (-np 1) and "replay" (= -np 4 default, which on his tree is
#     the plane-indexed snapshots: there is no replay before v1.1.0). Question: is the -np 4 text equal to the -np 1 text
#     on HIS tree for these requests? On the merge it is not for the snapshot arm (rs-modes-merge).
#  2. rs-modes2.sh on the merge build (its reading rule is in its header).
set -u
W=/root/rebase-v110; export RT=/opt/rocm/core-10.0
P=$W/queue-rs.progress
log(){ echo "$(date -Is) [queue-rs] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is running"; exit 1; }
pgrep -x llama-perplexit >/dev/null && { log "FATAL: a perplexity run is in progress"; exit 1; }
ARMS="ring replay" bash $W/rs-modes.sh /root/build-mxxm-b12500 $W/rs-modes-mxxm > $W/rs-modes-mxxm.out 2>&1; log "rs-modes on his master rc=$?"
bash $W/rs-modes2.sh ${B:-/root/build-merge-v110} $W/rs-modes2-${T:-merge} > $W/rs-modes2-${T:-merge}.out 2>&1; log "rs-modes2 rc=$?"
log "=== QUEUE DONE ==="
