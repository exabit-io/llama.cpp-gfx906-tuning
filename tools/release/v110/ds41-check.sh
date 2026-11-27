#!/bin/bash
# ds41-check.sh -- build and test the DeepSeek-V4.1 upstream-layout port (branch wip/ds41-upstream-layout, /root/wt-ds41:
# upstream v1.1.0's loader and graph as a second class beside mxxm-t's, chosen by the file's keys; commit by a
# subagent, not built before this). Correctness only (dies at 1000 MHz): no rate printed here is a number.
# READING, before the run:
#  1. The build has 0 errors.
#  2. test-llama-archs -a deepseek41 passes (the upstream fixture, skipped on his line).
#  3. test-llama-context over every fixture model: the two KV-replica models fail as on master (gemma4-dense,
#     mistral-large4-moe) and NOTHING else does; the count of models may grow by the deepseek41 fixture.
#  4. The model on this box (upstream layout, 1043 tensors) loads; -sm layer greedy text is coherent; perplexity
#     2K x 4 is within its own error of the 1.0.0 package's 2.2164 +/- 0.0595 (upstream's numerics moved a little
#     between v1.0.0 and v1.1.0, so equality is not expected); -sm tensor greedy text is coherent; the server answers.
#  5. Reported, not judged: the DSpark sidecar's accepted/drafted (1.0.0: 115/227).
# Anything else stops the port: it is not merged into master until explained.
# TAG in the environment names the output files of a rerun (default ds41; ds41b = the rerun on the padding fix).
set -u
W=/root/rebase-v110; RT=/opt/rocm/core-10.0; SRC=/root/wt-ds41; B=/root/build-ds41; T=${TAG:-ds41}; P=$W/$T-check.progress
log(){ echo "$(date -Is) [ds41] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is running"; exit 1; }
log "source $(git -C $SRC rev-parse --short HEAD), $(git -C $SRC status --porcelain --untracked-files=no | wc -l) uncommitted"
if ! UIVER=b12774 bash $W/build-rocm.sh $SRC $B $W/build-$T.log $RT > $W/build-$T.out 2>&1; then log "1. BUILD FAILED: $(grep -m3 ' error: ' $W/build-$T.log | tr '\n' ' ' | cut -c1-400)"; exit 1; fi
log "1. build: $(grep 'BUILD rc' $W/build-$T.out | cut -c1-80)"
LD_LIBRARY_PATH=$B/bin:$RT/lib HSA_FORCE_FINE_GRAIN_PCIE=1 timeout 1800 $B/bin/test-llama-archs -a deepseek41 > $W/$T-archs.out 2>&1
log "2. test-llama-archs -a deepseek41 rc=$?: $(sed 's/\x1b\[[0-9;]*m//g' $W/$T-archs.out | grep -i -E 'pass|fail|skip|error' | tail -3 | tr '\n' ';' | cut -c1-300)"
SCLK_GUARD=0 RT=$RT bash $W/ctx-check.sh $B $W/ctx-$T > $W/ctx-$T.out 2>&1
log "3. ctx-check rc=$?: $(grep 'test-llama-context' $W/ctx-$T/progress | cut -c40-420)"
log "3. failing models: $(sed 's/\x1b\[[0-9;]*m//g' $W/ctx-$T/context.out 2>/dev/null | grep FAIL | awk '{print $1}' | sort -u | tr '\n' ' ')"
SCLK_GUARD=0 RT=$RT bash $W/dsv41-test.sh $B $W/dsv41-$T full > $W/dsv41-$T.out 2>&1
log "4. dsv41-test rc=$?"
grep -v 'clock\|sampler' $W/dsv41-$T/progress | cut -c27-330 | while read -r l; do log "4. $l"; done
log "=== DS41-CHECK DONE ==="
