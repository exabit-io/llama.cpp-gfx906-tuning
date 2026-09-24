#!/bin/bash
# C5 gate on gfx906-substrate-v041: perplexity 16K/6 after the all-dies test-backend-ops.
# Methodology copied from the corpus so the number is comparable:
#   -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f wiki.test.raw, tp4 tensor split
# Reference: PPL = 5.6171 +/- 0.06236 (the 7.14-vs-10.0 gate, bit-identical over 4 rotated runs).
# Historical corpus cluster: 5.5969 - 5.6448. A result inside that band passes R3.5.
set -u
W=/root/night-20260919; B=/root/build-substrate-v041
M=/root/models/Qwen3.8-27B-Q8_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
/bin/bash $W/waitproc.sh "${1:-}" "^$B/bin/test-backend-ops" || echo "$(date -Is) waitproc failed ($?)" >> $W/chain.log
echo "$(date -Is) test-backend-ops finished; starting perplexity 16K/6 gate" >> $W/chain.log
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
export LD_LIBRARY_PATH=$B/bin:$B/lib:/opt/rocm/core-10.0/lib
timeout 7200 $B/bin/llama-perplexity -m $M \
  --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on \
  -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $W/ppl-substrate-v041.log 2>&1
rc=$?
echo "$(date -Is) perplexity exit=$rc" >> $W/chain.log
grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $W/ppl-substrate-v041.log >> $W/chain.log 2>/dev/null
echo "$(date -Is) === C5 GATE DONE ===" >> $W/chain.log
