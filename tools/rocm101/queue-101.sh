#!/bin/bash
# queue-101.sh -- after the correctness suite: package tests of the two ROCm 10.1 debs, then the 10.0-vs-10.1 A/B at the
# design points (rocm-ab.sh, ~3.2 h). Nothing else may compile or measure on the host while this runs.
set -u
W=/root/rocm101-migration-20261114; Q=$W/queue-101.progress
log(){ echo "$(date -Is) [queue] $*" | tee -a $Q; }
until grep -q 'SUITE DONE' $W/suite-101.progress 2>/dev/null; do sleep 15; done
log "start; load $(cut -d' ' -f1-3 /proc/loadavg)"
bash /root/pytorch-gfx906-rocm10.1-2.15.0/deb-test.sh /root/pytorch-gfx906-rocm10.1-2.15.0/deb-test > $W/deb-test-torch-101.out 2>&1; log "torch deb-test rc=$?"
bash $W/deb-test-llama-101.sh $W/deb-test-llama-101 > $W/deb-test-llama-101.out 2>&1; log "llama deb-test rc=$?"
bash $W/rocm-ab.sh > $W/rocm-ab.out 2>&1; log "rocm-ab rc=$?"
log "=== QUEUE DONE ==="
