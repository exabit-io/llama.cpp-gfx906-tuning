#!/bin/bash
# tg-test.sh PREFIX [LABEL] [extra llama-bench args]: tp4 tg128 x5 with per-sample tok/s (llama-bench -o json), gfx906.env; appends to bisect/tg.md
P=$1; L=${2:-$(basename $P)}; shift 2 2>/dev/null; B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib
timeout 600 $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 0 -n 128 -r 5 "$@" -o json > $B/bisect/tg-$L.json 2>$B/bisect/tg-$L.err
python3 - "$B/bisect/tg-$L.json" "$L" >> $B/bisect/tg.md <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))[0]; s = [128 / (x / 1e9) for x in d['samples_ns']]
    print(f"| {sys.argv[2]} | {d['avg_ts']:.1f} ± {d['stddev_ts']:.1f} | " + ' '.join(f"{x:.1f}" for x in s) + " |")
except Exception as e:
    print(f"| {sys.argv[2]} | fail | {e} |")
PY
tail -1 $B/bisect/tg.md
