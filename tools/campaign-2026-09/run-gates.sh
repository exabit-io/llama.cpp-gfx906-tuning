#!/bin/bash
# C5 perplexity gate, then Part 6 gate 3 (variance). Sequential in ONE script: no pgrep-based
# waiting anywhere. T4 variant learned 2026-09-20: a wait loop keyed on `pgrep -f <pattern>` can
# match ANY shell whose command text mentions the pattern — including the monitoring shell itself —
# and then never exits. Chain by sequence or by PID, never by pattern.
set -u
W=/root/night-20260919; B=/root/build-substrate-v041
M=/root/models/Qwen3.8-27B-Q8_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
export LD_LIBRARY_PATH=$B/bin:$B/lib:/opt/rocm/core-10.0/lib
echo "$(date -Is) starting C5 perplexity gate 16K/6" >> $W/chain.log
timeout 7200 $B/bin/llama-perplexity -m $M \
  --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on \
  -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $W/ppl-substrate-v041.log 2>&1
echo "$(date -Is) perplexity exit=$? " >> $W/chain.log
grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $W/ppl-substrate-v041.log >> $W/chain.log 2>/dev/null
echo "$(date -Is) starting Part 6 gate 3: variance" >> $W/chain.log
/bin/bash $W/variance-gate.sh >> $W/variance.out 2>&1
echo "$(date -Is) variance exit=$? " >> $W/chain.log
echo "$(date -Is) === GATES DONE ===" >> $W/chain.log
