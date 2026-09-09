#!/bin/bash
# m1-trace.sh - NEXT-STEPS M1: kernel trace of decode tokens on tp4 (and on one die), production build (mxxm-fh).
# Runs: unprofiled references (tp4, die0) -> tp4 kernel trace -> tp4 kernel + HIP runtime + RCCL API trace -> die0 kernel trace.
# Output: /root/rocm-tests/bench/trace-m1/  (m1.progress is the log; .done marks completion)
set -u
B=/root/rocm-tests/bench; OUT=$B/trace-m1; PROG=$OUT/m1.progress
. $B/gpu-test-env.sh
M=/root/models/Qwen3.8-27B-Q8_0.gguf
P=/opt/llama.cpp-mxxm-fh
export LD_LIBRARY_PATH=$P/lib
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
log(){ echo "$(date -Is) $*" | tee -a "$PROG"; }
SAMP=""; SMCPID=""
cleanup(){ restore; [ -n "$SAMP" ] && kill $SAMP 2>/dev/null; [ -n "$SMCPID" ] && kill $SMCPID 2>/dev/null; }
trap cleanup EXIT
log "M1 start pid $$"
setmax
start_sampler $OUT/clocks.log
nohup $B/smc-log.sh $OUT/smc.log >/dev/null 2>&1 & SMCPID=$!
log "state: $(state_line)"
log "libs: $(ldd $P/bin/llama-bench | grep -o '/opt/[^ ]*libggml-hip[^ ]*' | head -1)"

log "smoke: rocprofv3 --kernel-trace on hbm-bw (die 0, 1 s, 256 MiB)"
timeout 120 rocprofv3 --kernel-trace --stats -f csv -d $OUT/smoke -o smoke -- $B/hbm-bw 0 1 268435456 > $OUT/smoke.out 2>&1 \
  || { log "SMOKE FAILED (rc $?), see $OUT/smoke.out"; exit 1; }
log "smoke ok: $(ls $OUT/smoke | tr '\n' ' ')"

TP4="-dev rocm0/rocm1/rocm2/rocm3 -sm tensor"
COMMON="-m $M -fa 1 -p 0 -n 64 -o md"

log "1/5 reference tp4 (no profiler, -r 3)"
$P/bin/llama-bench $COMMON $TP4 -r 3 > $OUT/ref-tp4.md 2> $OUT/ref-tp4.err; grep -E '^\|' $OUT/ref-tp4.md | tail -1 >> "$PROG"
log "2/5 reference die0 (no profiler, -r 3)"
$P/bin/llama-bench $COMMON -dev rocm0 -r 3 > $OUT/ref-die0.md 2> $OUT/ref-die0.err; grep -E '^\|' $OUT/ref-die0.md | tail -1 >> "$PROG"

log "3/5 tp4 kernel trace"
timeout 900 rocprofv3 --kernel-trace --stats -f csv -d $OUT/tp4-kt -o tp4-kt -- $P/bin/llama-bench $COMMON $TP4 -r 1 > $OUT/tp4-kt.md 2> $OUT/tp4-kt.err
log "  rc $?; $(grep -E '^\|' $OUT/tp4-kt.md | tail -1)"

log "4/5 tp4 kernel + HIP runtime + RCCL API trace"
timeout 900 rocprofv3 --kernel-trace --hip-runtime-trace --rccl-trace --stats -f csv -d $OUT/tp4-api -o tp4-api -- $P/bin/llama-bench $COMMON $TP4 -r 1 > $OUT/tp4-api.md 2> $OUT/tp4-api.err
log "  rc $?; $(grep -E '^\|' $OUT/tp4-api.md | tail -1)"

log "5/5 die0 kernel trace"
timeout 900 rocprofv3 --kernel-trace --stats -f csv -d $OUT/die0-kt -o die0-kt -- $P/bin/llama-bench $COMMON -dev rocm0 -r 1 > $OUT/die0-kt.md 2> $OUT/die0-kt.err
log "  rc $?; $(grep -E '^\|' $OUT/die0-kt.md | tail -1)"

log "DONE"; touch $OUT/.done
