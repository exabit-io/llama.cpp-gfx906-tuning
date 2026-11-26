#!/bin/bash
# ctx-check.sh BUILD OUTDIR -- the model-fixture tests of the tree, run directly (ctest sets GGML_SCHED_DEBUG_REALLOC=1 on
# test-llama-context, which stops on the fork's scheduler reallocations; that predates the merges):
#  1. test-llama-archs: builds a dummy model of every architecture and runs it
#  2. test-llama-context over those models, one worker
#  3. test-fp8 (requantize, KV cache) over the FP8 dummy models
set -u
B=$1; O=$2
mkdir -p $O; P=$O/progress
LDP=$B/bin:${RT:-/opt/rocm/core-10.1}/lib
log(){ echo "$(date -Is) [ctx] $*" | tee -a $P; }
export HSA_FORCE_FINE_GRAIN_PCIE=1
MD=$O/test-models; FD=$O/test-models-fp8; rm -rf $MD $FD $O/logs; mkdir -p $MD $FD   # the test creates its log directory itself
LD_LIBRARY_PATH=$LDP timeout 3600 $B/bin/test-llama-archs -o $MD/ > $O/archs.out 2> $O/archs.err < /dev/null
log "test-llama-archs rc=$? models=$(ls $MD | wc -l); $(sed 's/\x1b\[[0-9;]*m//g' $O/archs.out $O/archs.err | grep -iE 'passed|failed|FAIL' | tail -3 | tr '\n' ' ' | cut -c1-300)"
LD_LIBRARY_PATH=$LDP timeout 3600 $B/bin/test-llama-context --models $MD/ --logs $O/logs -j 1 > $O/context.out 2> $O/context.err < /dev/null
log "test-llama-context rc=$? $(sed 's/\x1b\[[0-9;]*m//g' $O/context.err $O/context.out | grep -E 'passed, .* failed' | awk '{f+=$(NF-3)} END {print NR" cases, failed total "f}'); failing: $(sed 's/\x1b\[[0-9;]*m//g' $O/context.err $O/context.out | grep -E 'passed, .* failed' | grep -v ' 0 failed' | sed 's/.*main: //' | tr '\n' ';' | cut -c1-300)"
if [ -x $B/bin/test-fp8 ]; then
  LD_LIBRARY_PATH=$LDP timeout 1800 $B/bin/test-llama-archs -a llama -s 1 -o $FD > $O/fp8-gen.out 2>&1 < /dev/null
  log "fp8 models rc=$? $(ls $FD | wc -l) files"
  LD_LIBRARY_PATH=$LDP timeout 1800 $B/bin/test-fp8 --models $FD > $O/fp8-requantize.out 2>&1 < /dev/null
  log "test-fp8 requantize rc=$? $(sed 's/\x1b\[[0-9;]*m//g' $O/fp8-requantize.out | grep -iE 'pass|fail|error' | tail -2 | tr '\n' ' ' | cut -c1-240)"
  LD_LIBRARY_PATH=$LDP timeout 1800 $B/bin/test-fp8 --models $FD --kv > $O/fp8-kv.out 2>&1 < /dev/null
  log "test-fp8 kv rc=$? $(sed 's/\x1b\[[0-9;]*m//g' $O/fp8-kv.out | grep -iE 'pass|fail|error' | tail -2 | tr '\n' ' ' | cut -c1-240)"
fi
log "=== CTX DONE ==="
