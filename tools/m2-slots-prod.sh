#!/bin/bash
# NEXT-STEPS M2: production build (/opt/llama.cpp-mxxm-fh) at 24 and 32 slots (MMQ tile kernel) and 12-16 at 8K/32K depth. tp4 llama-batched-bench.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-mxxmfh-m2-slots; P=/opt/llama.cpp-mxxm-fh
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md
echo "# $TAG  $(date -Is)  production build tp4 batched-bench: slots 8-32 at 2K, 12-32 at 8K/32K (libggml-hip: $(ldd $P/bin/llama-batched-bench | grep -o '/opt/[^ ]*libggml-hip[^ ]*'))" > $OUT
{ echo; echo "## 2K: -npp 2048 -ntg 128 -npl 8,12,16,24,32 -c 69632"; $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 8,12,16,24,32 2>$B/$TAG-2k.err | grep -E '^\|'; echo "(exit ${PIPESTATUS[0]})"; } >> $OUT
log "2K done: $(grep -E '^\| *2048 ' $OUT | awk -F'|' '{gsub(/ +/,""); printf "%s:%s ", $4, $9}')"
{ echo; echo "## depth: -npp 8192,32768 -ntg 128 -npl 12,16,24,32 -c 1052672"; $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 1052672 -npp 8192,32768 -ntg 128 -npl 12,16,24,32 2>$B/$TAG-depth.err | grep -E '^\|'; echo "(exit ${PIPESTATUS[0]})"; } >> $OUT
log "depth done: $(grep -E '^\| *(8192|32768) ' $OUT | awk -F'|' '{gsub(/ +/,""); printf "%s@%s:%s ", $4, $2, $9}')"
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
