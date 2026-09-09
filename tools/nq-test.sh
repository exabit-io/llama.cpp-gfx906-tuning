#!/bin/bash
# S3 step 1 check: the norm + q8_1 fusion build (/opt/llama.cpp-mxxm-fh-nq) against the production build (/opt/llama.cpp-mxxm-fh).
#  A. greedy completion text must be identical (or perplexity identical to 4 digits if it differs by late drift)
#  B. perplexity tp4 -c 16384 --chunks 6 (reference 5.5969)
#  C. llama-bench pp2048 + tg128 (-r 3) on tp4 and one die, both builds, interleaved
#  D. batched-bench tp4 8/16 slots at 2K, both builds
#  E. kernel trace of 64 tokens on tp4 (nq build): kernels per token and quantize count (M1 method)
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-nq-test; NQ=/opt/llama.cpp-mxxm-fh-nq; PR=/opt/llama.cpp-mxxm-fh
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
[ -x $NQ/bin/llama-bench ] || { log "no $NQ build"; exit 1; }
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md; F=/root/models/wikitext-2-raw/wiki.test.raw
PROMPT="The four dies of a Radeon Pro Vega II Duo share one XGMI ring. In a tensor split, each decode token"
echo "# $TAG  $(date -Is)  norm+q8_1 fusion build vs production build" > $OUT
# A. greedy text
for P in $NQ $PR; do n=$(basename $P); LD_LIBRARY_PATH=$P/lib $P/bin/llama-completion -m $M $D4 -fa on -b 2048 -ub 2048 -c 4096 -n 200 --temp 0 --seed 1 -no-cnv -p "$PROMPT" > $B/$TAG-$n-greedy.txt 2>$B/$TAG-$n-greedy.err; log "$n greedy: $(grep -c . $B/$TAG-$n-greedy.txt) lines; fusion log: $(grep -o 'rms_norm + q8_1 fusion active[^)]*)' $B/$TAG-$n-greedy.err | head -1)"; done
if cmp -s $B/$TAG-$(basename $NQ)-greedy.txt $B/$TAG-$(basename $PR)-greedy.txt; then echo "A. greedy 200-token completion: IDENTICAL text on both builds" >> $OUT; else echo "A. greedy completion DIFFERS (see $TAG-*-greedy.txt; first differing line: $(diff $B/$TAG-$(basename $NQ)-greedy.txt $B/$TAG-$(basename $PR)-greedy.txt | head -3 | tr '\n' ' ' | cut -c1-200))" >> $OUT; fi
log "A: $(tail -1 $OUT | cut -c1-100)"
# B. perplexity
LD_LIBRARY_PATH=$NQ/lib $NQ/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $B/$TAG-ppl.log 2>&1
fe=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-ppl.log | tail -1); echo "B. perplexity nq build tp4 -c 16384 --chunks 6: ${fe:-none} (reference 5.5969 +/- 0.062)" >> $OUT; log "B: ${fe:-none}"
# C. llama-bench, interleaved
{ echo; echo "## C. llama-bench -p 2048 -n 128 -r 3, tp4 and rocm0: nq (both fusions), prod, nq with the q8 fusion off, with the add-norm fusion off, with the GDN producer fusion off, nq again"; echo "| build | device | test | t/s |"; echo "|---|---|---|---:|"; } >> $OUT
lb() { local P=$1 n=$2; shift 2; env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3,rocm0 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 128 -d 0 -r 3 -o md 2>$B/$TAG-$n-lb.err | grep -E '^\| qwen' | awk -F'|' -v n=$n '{gsub(/^ +| +$/,"",$9); gsub(/^ +| +$/,"",$10); gsub(/^ +| +$/,"",$11); printf "| %s | %s | %s | %s |\n", n, $9, $10, $11}' >> $OUT; log "$n lb: $(tail -4 $OUT | awk -F'|' '{printf "%s %s %s; ", $3, $4, $5}') fusions: $(grep -oE '(q8_1|add \+ rms_norm) fusion active' $B/$TAG-$n-lb.err | sort -u | tr '\n' ' ')"; }
lb $NQ nq; lb $PR prod; lb $NQ nq-noq8 GGML_CUDA_NORM_Q8=0; lb $NQ nq-noaddnorm GGML_CUDA_ADD_NORM=0; lb $NQ nq-nogdn GGML_CUDA_GDN_PREFUSE=0; lb $NQ nq-repeat
# D. batched
{ echo; echo "## D. batched-bench tp4 -npp 2048 -ntg 128 -npl 8,16"; } >> $OUT
for P in $NQ $PR; do n=$(basename $P); { echo "### $n"; LD_LIBRARY_PATH=$P/lib $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 34816 -npp 2048 -ntg 128 -npl 8,16 2>$B/$TAG-$n-bb.err | grep -E '^\| *2048 '; } >> $OUT; log "$n bb: $(grep -E '^\| *2048 ' $OUT | tail -2 | awk -F'|' '{gsub(/ /,"",$4); gsub(/ /,"",$9); printf "%s:%s ", $4, $9}')"; done
# E. trace
LD_LIBRARY_PATH=$NQ/lib timeout 900 rocprofv3 --kernel-trace -f csv -d $B/trace-nq -o nq -- $NQ/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 0 -n 64 -r 1 -o md > $B/$TAG-trace.md 2>$B/$TAG-trace.err
{ echo; echo "## E. kernel trace, nq build, tp4 single stream (M1 method; production: 1866 kernels per token per die, 257 quantize_q8_1 + 176 k_bin_bcast adds, busy 20.4 ms)"; echo '```'; python3 $B/m1-analyze.py $B/trace-nq nq --ref-ms 21.5 --gib-per-die 6.3 --show-token 3 2>&1 | grep -E '^== Agent 1|kernels [0-9]|mmvq:|weight|unprofiled|quantize|rms_norm|k_bin_bcast|l2_norm|sigmoid|softplus' | head -16; echo '```'; } >> $OUT
log "E: $(grep -E 'kernels [0-9]' $OUT | head -1)"
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
