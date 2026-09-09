#!/bin/bash
# Which part of the GDN producer fold changes the numerics (queue-15 check: greedy text differs at char 556, PPL 5.6083 vs 5.5969):
# masks 1 = q/k L2 norms, 2 = beta sigmoid, 4 = gate chain (did not apply). Greedy vs production, perplexity per mask.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-nq-gdnsub; NQ=/opt/llama.cpp-mxxm-fh-nq; PR=/opt/llama.cpp-mxxm-fh
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md; F=/root/models/wikitext-2-raw/wiki.test.raw
PROMPT="The four dies of a Radeon Pro Vega II Duo share one XGMI ring. In a tensor split, each decode token"
echo "# $TAG  $(date -Is)  GDN fold sub-masks vs production (restricted fusion build)" > $OUT
{ echo "| mask | meaning | greedy differs at char | PPL (ref 5.5969) |"; echo "|---|---|---:|---|"; } >> $OUT
LD_LIBRARY_PATH=$PR/lib $PR/bin/llama-completion -m $M $D4 -fa on -b 2048 -ub 2048 -c 4096 -n 200 --temp 0 --seed 1 -no-cnv -p "$PROMPT" > $B/$TAG-prod-greedy.txt 2>/dev/null
for mv in "1 q/k-l2-only" "2 beta-sigmoid-only" "7 all" "0 off"; do set -- $mv; mask=$1; name=$2
  GGML_CUDA_GDN_PREFUSE=$mask LD_LIBRARY_PATH=$NQ/lib $NQ/bin/llama-completion -m $M $D4 -fa on -b 2048 -ub 2048 -c 4096 -n 200 --temp 0 --seed 1 -no-cnv -p "$PROMPT" > $B/$TAG-m$mask-greedy.txt 2>/dev/null
  pos=$(cmp $B/$TAG-m$mask-greedy.txt $B/$TAG-prod-greedy.txt 2>/dev/null | grep -oE 'byte [0-9]+' | grep -oE '[0-9]+'); [ -z "$pos" ] && pos=-1
  GGML_CUDA_GDN_PREFUSE=$mask LD_LIBRARY_PATH=$NQ/lib $NQ/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $B/$TAG-m$mask-ppl.log 2>&1
  fe=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-m$mask-ppl.log | tail -1 | sed 's/Final estimate: PPL = //')
  echo "| $mask | $name | $pos | ${fe:-none} |" >> $OUT; log "mask $mask ($name): greedy $pos ppl ${fe:-none}"
done
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
