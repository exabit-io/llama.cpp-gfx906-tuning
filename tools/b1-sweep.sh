#!/bin/bash
# NEXT-STEPS S6: batch-1 MMVQ variants (rows / warps per block at one column, whole-block load at one column) on the fusion tree.
# llama-bench -p 0 -n 128 -r 3 on tp4 and rocm0 for each build; the baseline (defaults) first and last.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-nq-b1-sweep
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
OUT=$B/$TAG.md
{ echo "# $TAG  $(date -Is)  batch-1 MMVQ variants, fusion tree; llama-bench -p 0 -n 128 -r 3"
  echo "| build | knobs (ROWS1 NWARPS1 VDR8_1COL) | tp4 tg128 | rocm0 tg128 |"; echo "|---|---|---:|---:|"; } > $OUT
run() { local P=$1 name=$2 knobs=$3; [ -x $P/bin/llama-bench ] || { echo "| $name | $knobs | missing | missing |" >> $OUT; return; }
  LD_LIBRARY_PATH=$P/lib $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3,rocm0 -sm tensor -fa 1 -p 0 -n 128 -r 3 -o md > $B/$TAG-$name.md 2>$B/$TAG-$name.err
  local t4 t0; t4=$(grep -E '^\| qwen' $B/$TAG-$name.md | grep 'ROCm0/' | awk -F'|' '{gsub(/^ +| +$/,"",$11); print $11}'); t0=$(grep -E '^\| qwen' $B/$TAG-$name.md | grep -v 'ROCm0/' | awk -F'|' '{gsub(/^ +| +$/,"",$11); print $11}')
  echo "| $name | $knobs | ${t4:-fail} | ${t0:-fail} |" >> $OUT; log "$name: tp4 $t4 die0 $t0"; }
run /opt/llama.cpp-mxxm-fh-nq  nq-base "1 2 0"
run /opt/llama.cpp-nq-b1-r2    r2      "2 2 0"
run /opt/llama.cpp-nq-b1-r4    r4      "4 2 0"
run /opt/llama.cpp-nq-b1-v8    v8      "1 2 1"
run /opt/llama.cpp-nq-b1-r2v8  r2v8    "2 2 1"
run /opt/llama.cpp-nq-b1-w4    w4      "1 4 0"
run /opt/llama.cpp-mxxm-fh     prod    "production build, no fusions"
run /opt/llama.cpp-mxxm-fh-nq  nq-base-repeat "1 2 0"
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
