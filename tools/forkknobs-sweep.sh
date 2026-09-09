#!/bin/bash
# NEXT-STEPS S2 shortcut (2026-09-08): the ML-gfx906 fork's OWN custom peer-write allreduce (GGML_ENABLE_CUSTOM_AR=1, needs
# HSA_FORCE_FINE_GRAIN_PCIE=1 on gfx906 for the broadcast kernel) and whole-token graph (GGML_META_TOKEN_GRAPH, on by default but only
# records on the custom AR), never measured on this box. tp4, production build: llama-bench pp2048 / tg128 (-r 2) and batched-bench
# 8 / 16 slots at 2K per variant; perplexity and a HIP-runtime trace (graph launches and AR kernels per token) for the peer-write variant.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-mxxmfh-forkknobs; P=/opt/llama.cpp-mxxm-fh
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md; F=/root/models/wikitext-2-raw/wiki.test.raw
{ echo "# $TAG  $(date -Is)  fork knobs on the production build, tp4 (libggml-hip: $(ldd $P/bin/llama-bench | grep -o '/opt/[^ ]*libggml-hip[^ ]*'))"
  echo "| variant | env | AR path (from the log) | pp2048 t/s | tg128 t/s | 8 slots 2K t/s | 16 slots 2K t/s |"; echo "|---|---|---|---:|---:|---:|---:|"; } > $OUT
variant() { local name=$1; shift
  log "$name start ($*)"
  env "$@" $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 128 -d 0 -r 2 -o md > $B/$TAG-$name-lb.md 2>$B/$TAG-$name-lb.err
  local pp tg path b8 b16
  pp=$(grep -E '^\| qwen.*pp2048' $B/$TAG-$name-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$11); print $11}')
  tg=$(grep -E '^\| qwen.*tg128' $B/$TAG-$name-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$11); print $11}')
  path=$(grep -oE 'TP custom AllReduce: initialized for [0-9]+ GPUs, path = [^(]*|TP custom AR: [a-z ]*' $B/$TAG-$name-lb.err | head -1 | sed 's/TP custom AllReduce: initialized for [0-9]* GPUs, path = //')
  env "$@" $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 34816 -npp 2048 -ntg 128 -npl 8,16 > $B/$TAG-$name-bb.md 2>$B/$TAG-$name-bb.err
  b8=$(grep -E '^\| *2048 ' $B/$TAG-$name-bb.md | awk -F'|' '$4+0==8 {gsub(/ /,"",$9); print $9}'); b16=$(grep -E '^\| *2048 ' $B/$TAG-$name-bb.md | awk -F'|' '$4+0==16 {gsub(/ /,"",$9); print $9}')
  echo "| $name | $* | ${path:-RCCL (no custom AR)} | ${pp:-fail} | ${tg:-fail} | ${b8:-fail} | ${b16:-fail} |" >> $OUT
  log "$name: pp $pp tg $tg b8 $b8 b16 $b16 [${path:-RCCL}]"; }
variant baseline            BENCH_VARIANT=baseline
variant custom-ar-oneshot   GGML_ENABLE_CUSTOM_AR=1
variant peerwrite           GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1
variant peerwrite-no-tg     GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_META_TOKEN_GRAPH=0
variant peerwrite-nogate    GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_NO_GATE=1
variant fgp-only            HSA_FORCE_FINE_GRAIN_PCIE=1
variant serial-dispatch     GGML_META_PARALLEL_DISPATCH=0
variant baseline-repeat     BENCH_VARIANT=baseline2
# correctness of the peer-write path
log "perplexity peerwrite start"
env GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 $P/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $B/$TAG-peerwrite-ppl.log 2>&1
fe=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-peerwrite-ppl.log | tail -1)
{ echo; echo "Perplexity, peer-write custom AR, tp4 -c 16384 --chunks 6: ${fe:-none} (reference 5.5969 +/- 0.062 on the same build with RCCL)"; } >> $OUT; log "ppl: ${fe:-none}"
# structure of one token on the peer-write path (graph launches, AR kernels): same analysis as M1
log "trace peerwrite start"
env GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 timeout 900 rocprofv3 --kernel-trace --hip-runtime-trace -f csv -d $B/trace-forkknobs -o peerwrite -- $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 0 -n 64 -r 1 -o md > $B/$TAG-trace.md 2>$B/$TAG-trace.err
{ echo; echo '```'; python3 $B/m1-analyze.py $B/trace-forkknobs peerwrite --api 2>&1 | grep -E '^== Agent 1|token period|kernels [0-9]|mmvq:|gaps per token|hipGraphLaunch|hipModuleLaunchKernel|hipStreamSynchronize|== hip_api' | head -12; echo '```'; } >> $OUT
log "trace: $(grep -E 'hipGraphLaunch' $OUT | head -1)"
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
