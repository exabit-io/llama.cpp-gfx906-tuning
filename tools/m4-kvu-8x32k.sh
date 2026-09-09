#!/bin/bash
# NEXT-STEPS M4: 8 x 32K decode on b10837, private slots vs --kv-unified pool (the one pooled-cache cell not re-measured; b10288: 130 private / 86 pool).
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-b10837-m4-kvu8x32k; NEW=/opt/llama.cpp-b10837
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$NEW/lib
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md
echo "# $TAG  $(date -Is)  b10837 tp4 batched-bench 8 x 32K, private vs --kv-unified (libggml-hip: $(ldd $NEW/bin/llama-batched-bench | grep -o '/opt/[^ ]*libggml-hip[^ ]*'))" > $OUT
for mode in private unified; do opt=""; [ $mode = unified ] && opt="--kv-unified"
  { echo; echo "## $mode: $D4 -fa on $opt -c 263168 -npp 32768 -ntg 128 -npl 8"; $NEW/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 $opt -c 263168 -npp 32768 -ntg 128 -npl 8 2>$B/$TAG-$mode.err | grep -E '^\|'; echo "(exit ${PIPESTATUS[0]})"; } >> $OUT
  log "$mode done: $(grep -E '^\| *32768 ' $OUT | tail -1 | cut -c1-120)"
done
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
