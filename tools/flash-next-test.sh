#!/bin/bash
# Qwen3.8-Flash-Next (qwen4exp, 512 experts, UD-Q4_K_XL 104 GiB) on the gfx906 branch build: four dies -sm tensor, PLE table sharded
# (LLAMA_PLE_SHARD=1, the fork's feature) and host-resident (0); llama-bench pp2048/tg128; a short perplexity; a greedy sample.
set -u
B=/root/rocm-tests/bench; TAG=qwen38-flash-next; P=/opt/llama.cpp-gfx906-master; MD=/root/models/Qwen3.8-Flash-Next-UD-Q4_K_XL; M=$MD/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
[ -f $MD/.download-done ] || { log "download not complete"; exit 1; }
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481 GPU_MAX_HW_QUEUES=8
LM=""; $P/bin/llama-bench --help 2>&1 | grep -q -- '--load-mode' && LM="--load-mode dio"
OUT=$B/$TAG.md; echo "# $TAG  $(date -Is)  $P, four dies -sm tensor, model $(du -sh $MD | cut -f1) ($LM)" > $OUT
{ echo; echo "## llama-bench -p 2048 -n 128 -r 2, tp4"; echo "| PLE shard | pp2048 | tg128 |"; echo "|---|---:|---:|"; } >> $OUT
for sh in 1 0; do LLAMA_PLE_SHARD=$sh timeout 3600 $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 128 -d 0 -r 2 $LM -o md > $B/$TAG-shard$sh-lb.md 2>$B/$TAG-shard$sh-lb.err
  cells=$(grep -E '^\| Qwen|^\| qwen' $B/$TAG-shard$sh-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$(NF-1)); printf "%s | ", $(NF-1)}'); echo "| $sh | ${cells:-fail: $(grep -m1 -iE 'error|failed|out of memory' $B/$TAG-shard$sh-lb.err | cut -c1-100)} |" >> $OUT; log "shard $sh: ${cells:-fail}"
  grep -m3 -iE 'PLE|ple' $B/$TAG-shard$sh-lb.err | cut -c1-160 >> $OUT
done
LLAMA_PLE_SHARD=1 timeout 3600 $P/bin/llama-perplexity -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -b 2048 -ub 2048 -c 2048 --chunks 4 $LM -f /root/models/wikitext-2-raw/wiki.test.raw > $B/$TAG-ppl.log 2>&1
fe=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-ppl.log | tail -1); { echo; echo "## perplexity -c 2048 --chunks 4 (PLE shard on): ${fe:-none (see $TAG-ppl.log)}"; } >> $OUT; log "ppl ${fe:-none}"
LLAMA_PLE_SHARD=1 timeout 900 $P/bin/llama-completion -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -c 4096 -n 120 --temp 0 -no-cnv $LM -p "Explain in three sentences why a tensor split over four GPUs needs an allreduce after every layer." > $B/$TAG-greedy.txt 2>$B/$TAG-greedy.err
{ echo; echo "## greedy sample (120 tokens)"; echo '```'; head -c 1200 $B/$TAG-greedy.txt; echo; echo '```'; } >> $OUT
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
