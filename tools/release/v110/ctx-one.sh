#!/bin/bash
# ctx-one.sh BUILD OUT MODELDIR MODEL... -- test-llama-context on a few of the dummy models test-llama-archs wrote into
# MODELDIR (ctx-check.sh keeps them in OUT/test-models), one worker, the ROCm runtime named by RT.
set -u
B=$1; O=$2; MD=$3; shift 3
mkdir -p $O/models; rm -f $O/models/*.gguf; rm -rf $O/logs   # the test creates its log directory itself
for m in "$@"; do ln -sf $MD/$m.gguf $O/models/$m.gguf; done
export HSA_FORCE_FINE_GRAIN_PCIE=1
LD_LIBRARY_PATH=$B/bin:${RT:-/opt/rocm/core-10.1}/lib timeout 1800 $B/bin/test-llama-context --models $O/models/ --logs $O/logs -j 1 > $O/context.out 2> $O/context.err < /dev/null
rc=$?
echo "$(date -Is) [ctx-one] $B rc=$rc models: $*; $(sed 's/\x1b\[[0-9;]*m//g' $O/context.err $O/context.out | grep -E 'passed, .* failed|fatal error|GGML_ASSERT' | tr '\n' ';' | tr -s ' ' | cut -c1-400)" | tee -a $O/progress
