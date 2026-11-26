#!/bin/bash
# numerics.sh OUT BUILD... -- did the numbers of Qwen3.8-27B Q8_0 move between two trees, and where? Per build (name of
# its directory), ROCm 10.0 runtime, four dies: perplexity 16K x 6 with -sm tensor and with -sm layer, and 192 greedy
# tokens with -sm tensor without any drafter (text hash). A perplexity is a function of the logits alone, so two builds
# with the same value to the last printed digit and the same text compute the same numbers on that path.
# Used on 2026-11-26 to attribute 5.6126 -> 5.6130 (mxxm-t b12500 -> the v1.1.0 merge): stock b12500 against stock
# v1.1.0 says whether upstream moved by itself. Not a timing run.
set -u
O=$1; shift
R=/root/rocm-tests/bench; RT=${RT:-/opt/rocm/core-10.0}
M=/root/models/Qwen3.8-27B-Q8_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
mkdir -p $O; P=$O/progress
log(){ echo "$(date -Is) [numerics] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
. $R/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
trap 'restore' EXIT
setmax
D4="--device rocm0,rocm1,rocm2,rocm3"
for B in "$@"; do
  n=$(basename $B); L=$B/bin:$B/lib:$RT/lib
  for sm in ${SMS:-tensor layer}; do
    LD_LIBRARY_PATH=$L timeout 3600 $B/bin/llama-perplexity -m $M $D4 -sm $sm -ngl all -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $O/$n-ppl-$sm.out 2> $O/$n-ppl-$sm.log < /dev/null
    log "$n ppl -sm $sm rc=$?: $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $O/$n-ppl-$sm.log $O/$n-ppl-$sm.out | tail -1 | sed 's/.*Final/Final/')"
  done
  LD_LIBRARY_PATH=$L timeout 900 $B/bin/llama-completion -m $M $D4 -sm tensor -ngl all -fa on -c 4096 -n 192 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup -p "Write a short paragraph about the history of the transistor." > $O/$n-gen.out 2> $O/$n-gen.log < /dev/null
  log "$n greedy 192 rc=$?: $(wc -c < $O/$n-gen.out) bytes sha $(sha256sum < $O/$n-gen.out | cut -c1-12)"
done
log "=== NUMERICS DONE ==="
