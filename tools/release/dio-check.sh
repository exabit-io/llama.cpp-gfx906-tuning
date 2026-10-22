#!/bin/bash
# dio-check.sh OUTDIR NAME=BUILD... -- does a -lm dio load fall back to buffered I/O? (the report on mxxm-t/mx-llama.cpp#17)
# Loads Qwen3.8-27B Q8_0 with -lm dio, -sm layer on four dies, generates 8 tokens, counts the fallback warnings.
set -u
O=$1; shift
R=/root/rocm-tests/bench
M=/root/models/Qwen3.8-27B-Q8_0.gguf
mkdir -p $O; P=$O/progress; DONE=$O/.done; rm -f $DONE; echo $$ > $O/pid
log(){ echo "$(date -Is) [dio] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
. $R/gpu-test-env.sh
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
BPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${BPID:-}"; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax
for arg in "$@"; do
  name=${arg%%=*}; B=${arg#*=}
  LD_LIBRARY_PATH=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib timeout 600 $B/bin/llama-completion -m $M \
    --device rocm0,rocm1,rocm2,rocm3 -sm layer -ngl 99 -fa on -lm dio -c 2048 -n 8 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup \
    -p "The capital of France is" > $O/$name.out 2> $O/$name.log < /dev/null &
  BPID=$!; wait $BPID; rc=$?; BPID=""
  log "$name rc=$rc fallbacks=$(grep -c 'Falling back to buffered' $O/$name.log) first='$(grep -m1 'Falling back' $O/$name.log | cut -c1-150)' load='$(grep -m1 -E 'load_tensors: .*(dio|direct)' $O/$name.log | cut -c1-120)' out='$(tr '\n' ' ' < $O/$name.out | cut -c1-80)'"
done
trap - INT TERM EXIT
restore; touch $DONE; log "=== DIO CHECK DONE ==="
