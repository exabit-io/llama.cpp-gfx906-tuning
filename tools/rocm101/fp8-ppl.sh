#!/bin/bash
# fp8-ppl.sh -- output quality with an FP8 KV cache (report s.5.12 left it unmeasured): perplexity and KL divergence
# against the f16 cache's logits, on the INSTALLED package, Qwen3.8-27B Q8_0, wikitext-2, 16K x 6 chunks, four dies
# -sm tensor (the package test's perplexity cell).
#   f16   -ctk f16     -ctv f16       reference; its logits are the KL base (written to a scratch file, removed at the end)
#   f8    -ctk f8_e4m3 -ctv f8_e4m3   the cache under test
#   q8    -ctk q8_0    -ctv q8_0      comparison: the compact cache this guide already calls quality-free (2026-09-06)
# READING, WRITTEN BEFORE THE FIRST RUN. Reported per arm: PPL, mean KLD, 99.9 % KLD, same-top-token share. The FP8 cache
# is called "quality-neutral" if its mean KLD is under 0.04 (optimize.py's --max-kl bound) AND not more than twice q8's;
# otherwise its values are reported as its cost. Whatever the outcome, section 5.12's throughput verdict stands
# (FP8 costs 47 % of 4 x 64K decode); this run only says whether the capacity lever is usable at all.
# 16K is under the 64K floor for SPEED results; this is a quality reading, as every perplexity cell of the guide is.
set -u
W=/root/rocm101-migration-20261114; O=$W/fp8-ppl; R=/root/rocm-tests/bench; B=/opt/mx-llama.cpp
M=/root/models/Qwen3.8-27B-Q8_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
NF=/root/llama.cpp-benchmarking/tools/assert-no-fa-fallback.sh
mkdir -p $O; P=$O/progress; BASE=$O/f16-logits.bin
log(){ echo "$(date -Is) [fp8ppl] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
[ -x $B/bin/llama-perplexity ] && [ -f $M ] && [ -f $F ] || { log "FATAL: no llama-perplexity, model or text"; exit 1; }
log "package $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906); free space $(df -BG --output=avail $O | tail -1 | tr -d ' ')"
. $R/gpu-test-env.sh
trap 'restore; rm -f $BASE' EXIT
setmax
export GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
run(){ # run NAME CTK CTV EXTRA...
  local n=$1 k=$2 v=$3 rc; shift 3
  timeout 7200 $B/bin/llama-perplexity -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -ctk $k -ctv $v \
    -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F "$@" > $O/$n.out 2> $O/$n.log < /dev/null; rc=$?
  local ppl kld fb
  ppl=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $O/$n.log $O/$n.out 2>/dev/null | tail -1 | sed 's/.*Final estimate: //')
  [ -n "$ppl" ] || ppl=$(grep -E 'Mean PPL\(Q\)' $O/$n.log $O/$n.out 2>/dev/null | tail -1 | tr -s ' ' | sed 's/^[^:]*://')
  kld=$(grep -hE 'Mean +KLD|99\.9% +KLD|Maximum KLD|Same top p|Mean +ln\(PPL\(Q\)/PPL\(base\)\)|Mean PPL\(base\)' $O/$n.log $O/$n.out 2>/dev/null | tr -s ' ' | tr '\n' ';')
  fb=$(bash $NF $O/$n.log > /dev/null 2>&1; echo $?)
  log "$n ($k/$v) rc=$rc: ${ppl:-NO PPL}; fallback check rc=$fb; ${kld:-no KLD lines}"
}
run f16 f16 f16 --kl-divergence-base $BASE
[ -s $BASE ] || { log "FATAL: no logits base written"; exit 1; }
log "base file $(du -h $BASE | cut -f1)"
run f8 f8_e4m3 f8_e4m3 --kl-divergence-base $BASE --kl-divergence
run q8 q8_0 q8_0 --kl-divergence-base $BASE --kl-divergence
log "=== FP8-PPL DONE ==="
