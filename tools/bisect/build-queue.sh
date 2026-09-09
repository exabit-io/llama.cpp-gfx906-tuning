#!/bin/bash
# Waits for the gfx906 validation's timing phases to end, then builds the perplexity-bisect candidates (CPU only, nice 19).
B=/root/rocm-tests/bench; L=$B/bisect/build-queue.log
while [ ! -f $B/.qwen38-27b-gfx906-master-validate-done ]; do sleep 30; done
echo "$(date -Is) validation done; building" >> $L
for sha in "$@"; do full=$(cd /root/upstream-bisect && git rev-parse --short $sha); echo "$(date -Is) build $full ($sha)" >> $L; $B/bisect/bisect-build.sh $full 20 >> $L 2>&1 && echo "$(date -Is) OK $full" >> $L || echo "$(date -Is) FAIL $full" >> $L; done
echo "$(date -Is) ALLBUILT" >> $L; touch $B/bisect/.builds-done
