#!/bin/bash
# Multi-stream graph optimisation on the tensor-split lanes (GGML_CUDA_GRAPH_OPT=2, fusion tree a1462e0): the small kernels are
# latency-bound and serialised on one stream per lane; independent branches (q/k/v matvecs, norms) could overlap.
# llama-bench pp2048/tg128 tp4 + rocm0, -r 3: off, =1 (single-device only: should equal off on tp4, on for rocm0), =2; then =2 with the
# adopted custom-AR settings; perplexity with =2 for correctness.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-nq-graphopt; NQ=/opt/llama.cpp-mxxm-fh-nq
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$NQ/lib
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md; F=/root/models/wikitext-2-raw/wiki.test.raw
AR="GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481"
{ echo "# $TAG  $(date -Is)  multi-stream graph optimisation on the split (fusion build)"; echo "| variant | env | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |"; echo "|---|---|---:|---:|---:|---:|"; } > $OUT
lb() { local name=$1; shift; env "$@" $NQ/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3,rocm0 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 128 -d 0 -r 3 -o md > $B/$TAG-$name-lb.md 2>$B/$TAG-$name-lb.err
  local cells; cells=$(grep -E '^\| qwen' $B/$TAG-$name-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$12); printf "%s | ", $12}'); echo "| $name | $* | $cells" >> $OUT; log "$name: $cells"; }
lb off      BENCH=off
lb opt1     GGML_CUDA_GRAPH_OPT=1
lb opt2     GGML_CUDA_GRAPH_OPT=2
lb ar       $AR
lb ar-opt2  $AR GGML_CUDA_GRAPH_OPT=2
lb off-rep  BENCH=off2
GGML_CUDA_GRAPH_OPT=2 $NQ/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $B/$TAG-opt2-ppl.log 2>&1
fe=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-opt2-ppl.log | tail -1); { echo; echo "Perplexity with GGML_CUDA_GRAPH_OPT=2 on tp4: ${fe:-none} (reference 5.5969)"; } >> $OUT; log "ppl opt2: ${fe:-none}"
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
