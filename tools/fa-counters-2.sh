#!/bin/bash
# TODO 9 / NEXT-STEPS S4, corrected pass: the first run (fa-counters.sh) asked for SQ_BUSY_CYCLES / SQ_WAIT_INST_ANY / TCP_TOTAL_CACHE_ACCESSES,
# which this ROCm does not define for gfx906 (only SQ_INSTS_VALU came back). This run uses the gfx906 set, three passes per phase:
#   A  GRBM_GUI_ACTIVE SQ_WAVES SQ_INSTS_VALU SQ_ACTIVE_INST_VALU        -> VALU busy (SQ_ACTIVE_INST_VALU*4/SIMDs/GRBM_GUI_ACTIVE), waves
#   B  SQ_INSTS_VMEM_RD SQ_INSTS_LDS SQ_WAIT_INST_LDS SQ_LDS_BANK_CONFLICT -> load and LDS mix, LDS stalls
#   C  TCC_HIT TCC_MISS SQ_INSTS_SALU SQ_INSTS_SMEM                       -> L2 hit rate (the KV re-read question), scalar share
# Workloads: prefill 4 x 16K on tp4 (same kernel, half the time); decode 4 x 32K via batched-bench (grid closer to the 4-sequence case).
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-fa-counters-2; P=/opt/llama.cpp-prod
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md; T=$B/trace-fa2; rm -rf $T; mkdir -p $T
echo "# $TAG  $(date -Is)  FA head-256 counters (gfx906 counter set), production build tp4" > $OUT
declare -A PMC=( [A]="GRBM_GUI_ACTIVE SQ_WAVES SQ_INSTS_VALU SQ_ACTIVE_INST_VALU" [B]="SQ_INSTS_VMEM_RD SQ_INSTS_LDS SQ_WAIT_INST_LDS SQ_LDS_BANK_CONFLICT" [C]="TCC_HIT TCC_MISS SQ_INSTS_SALU SQ_INSTS_SMEM" )
for pass in A B C; do
  log "prefill pass $pass: ${PMC[$pass]}"
  timeout 900 rocprofv3 --pmc ${PMC[$pass]} --kernel-trace -f csv -d $T/prefill-$pass -o pre -- $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 69632 -npp 16384 -ntg 1 -npl 4 > $B/$TAG-prefill-$pass.md 2>$B/$TAG-prefill-$pass.err; log "prefill $pass rc $?"
  log "decode pass $pass"
  timeout 900 rocprofv3 --pmc ${PMC[$pass]} --kernel-trace -f csv -d $T/decode-$pass -o dec -- $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 135168 -npp 32768 -ntg 16 -npl 4 > $B/$TAG-decode-$pass.md 2>$B/$TAG-decode-$pass.err; log "decode $pass rc $?"
done
python3 $B/fa-counters-2-analyze.py $T >> $OUT 2>&1
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
