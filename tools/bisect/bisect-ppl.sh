#!/bin/bash
# bisect-ppl.sh PREFIX [LABEL]: perplexity 16K / 6 chunks, tp4, the validation's exact settings; appends to bisect/ppl.md
P=$1; L=${2:-$(basename $P)}; B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib
timeout 900 $P/bin/llama-perplexity -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $B/bisect/ppl-$L.log 2>&1
fe=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/bisect/ppl-$L.log | tail -1)
echo "| $L | $(cd /root/upstream-bisect && git log -1 --format='%ad %s' --date=short ${L} 2>/dev/null | cut -c1-90) | ${fe:-FAIL} |" >> $B/bisect/ppl.md; echo "$L ${fe:-FAIL}"
