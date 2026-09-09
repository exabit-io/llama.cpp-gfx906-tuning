#!/bin/bash
# Fusion build follow-up: (1) which fusion changes the numerics (greedy text vs production, per knob), (2) perplexity of the
# all-off build (build parity) and of the differing knob, (3) llama-bench with the raw md saved (the first test lost the t/s column),
# (4) batched 8/16 with all fusions off (the first test showed prefill -8% on the fusion build).
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-nq-check2; NQ=/opt/llama.cpp-mxxm-fh-nq; PR=/opt/llama.cpp-mxxm-fh
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md; F=/root/models/wikitext-2-raw/wiki.test.raw
PROMPT="The four dies of a Radeon Pro Vega II Duo share one XGMI ring. In a tensor split, each decode token"
ALLOFF="GGML_CUDA_NORM_Q8=0 GGML_CUDA_ADD_NORM=0 GGML_CUDA_GDN_PREFUSE=0"
echo "# $TAG  $(date -Is)  fusion build follow-up" > $OUT
greedy() { local P=$1 name=$2; shift 2; env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-completion -m $M $D4 -fa on -b 2048 -ub 2048 -c 4096 -n 200 --temp 0 --seed 1 -no-cnv -p "$PROMPT" > $B/$TAG-$name-greedy.txt 2>$B/$TAG-$name-greedy.err; }
{ echo; echo "## A. greedy 200 tokens vs production (first differing character position; -1 = identical)"; echo "| variant | env | differs at char |"; echo "|---|---|---:|"; } >> $OUT
greedy $PR prod BENCH=prod
for v in "nq BENCH=nq" "noq8 GGML_CUDA_NORM_Q8=0" "noaddnorm GGML_CUDA_ADD_NORM=0" "nogdn GGML_CUDA_GDN_PREFUSE=0" "alloff $ALLOFF" "q8only GGML_CUDA_ADD_NORM=0 GGML_CUDA_GDN_PREFUSE=0" "addnormonly GGML_CUDA_NORM_Q8=0 GGML_CUDA_GDN_PREFUSE=0" "gdnonly GGML_CUDA_NORM_Q8=0 GGML_CUDA_ADD_NORM=0"; do set -- $v; name=$1; shift
  greedy $NQ $name "$@"
  pos=$(cmp $B/$TAG-$name-greedy.txt $B/$TAG-prod-greedy.txt 2>/dev/null | grep -oE 'byte [0-9]+' | grep -oE '[0-9]+'); [ -z "$pos" ] && pos=-1
  echo "| $name | $* | $pos |" >> $OUT; log "greedy $name: differs at ${pos}"
done
{ echo; echo "## B. perplexity tp4 -c 16384 --chunks 6 (reference 5.5969 +/- 0.062 on production)"; echo "| variant | PPL |"; echo "|---|---|"; } >> $OUT
for v in "alloff $ALLOFF" "q8only GGML_CUDA_ADD_NORM=0 GGML_CUDA_GDN_PREFUSE=0" "addnormonly GGML_CUDA_NORM_Q8=0 GGML_CUDA_GDN_PREFUSE=0" "gdnonly GGML_CUDA_NORM_Q8=0 GGML_CUDA_ADD_NORM=0"; do set -- $v; name=$1; shift
  env "$@" LD_LIBRARY_PATH=$NQ/lib $NQ/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $B/$TAG-$name-ppl.log 2>&1
  fe=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-$name-ppl.log | tail -1 | sed 's/Final estimate: PPL = //'); echo "| $name | ${fe:-none} |" >> $OUT; log "ppl $name: ${fe:-none}"
done
{ echo; echo "## C. llama-bench -p 2048 -n 128 -r 3 (raw md kept as $TAG-<variant>-lb.md)"; echo "| variant | env | tp4 pp2048 | tp4 tg128 | die0 pp2048 | die0 tg128 |"; echo "|---|---|---:|---:|---:|---:|"; } >> $OUT
lb() { local P=$1 name=$2; shift 2; env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3,rocm0 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 128 -d 0 -r 3 -o md > $B/$TAG-$name-lb.md 2>$B/$TAG-$name-lb.err
  local cells; cells=$(grep -E '^\| qwen' $B/$TAG-$name-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$12); printf "%s | ", $12}'); echo "| $name | $* | $cells" >> $OUT; log "lb $name: $cells"; }
lb $PR prod BENCH=prod; lb $NQ nq BENCH=nq; lb $NQ alloff $ALLOFF; lb $NQ noq8 GGML_CUDA_NORM_Q8=0; lb $NQ noaddnorm GGML_CUDA_ADD_NORM=0; lb $NQ nogdn GGML_CUDA_GDN_PREFUSE=0; lb $PR prod-repeat BENCH=prod2; lb $NQ nq-repeat BENCH=nq2
{ echo; echo "## D. batched-bench tp4 -npp 2048 -ntg 128 -npl 8,16, fusion build with all fusions off (first test: nq 1048/1068 pp vs prod 1137/1146)"; env $ALLOFF LD_LIBRARY_PATH=$NQ/lib $NQ/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 34816 -npp 2048 -ntg 128 -npl 8,16 2>$B/$TAG-alloff-bb.err | grep -E '^\| *2048 '; } >> $OUT
log "bb alloff: $(grep -E '^\| *2048 ' $OUT | tail -2 | awk -F'|' '{gsub(/ /,"",$4); gsub(/ /,"",$7); gsub(/ /,"",$9); printf "%s: pp %s gen %s; ", $4, $7, $9}')"
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
