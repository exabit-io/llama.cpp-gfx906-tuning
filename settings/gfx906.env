# Environment for llama.cpp on the four-die gfx906 box (macpro2019-01).
# Source this before llama-server / llama-bench:   set -a; . settings/gfx906.env; set +a
# Every line here is backed by a measurement in reports/ (see README.md, "Builds and settings").

# RCCL: give it the physical XGMI ring (firmware mislabels the bridge pairs) and 16 channels.
#   +1.8% prefill, +3.7% single-stream decode; server +3% at 1 client, +2% at 8, nothing at 4; nothing at batch 8 decode.
#   The only environment variable that moves anything (run-through s.1: nineteen others within 0.8% of a 0.3%-spread baseline, or losers).
NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml
NCCL_MIN_NCHANNELS=16

# HIP graphs stay ON (default). Turning them off costs 7% of single-stream decode.
# GGML_CUDA_DISABLE_GRAPHS=1     <- never

# The tensor-split allreduce stays on RCCL (default). llama.cpp's internal allreduce and the
# generic reduce are 23% slower in decode and 20% slower in prefill, with or without GGML_CUDA_P2P.
# GGML_CUDA_ALLREDUCE=internal   <- never
# GGML_CUDA_P2P=1                <- no effect (-0.3%)

# Measured and NOT adopted (run-through s.1, four interleaved baselines at 47.6 tok/s, 0.3% spread):
# NCCL_PROTO=LL                  bench-only batch-8 trade (+5% decode: 164.9 vs 157.0, -4.3% prefill); in the server -0.4% aggregate, +3% TTFT
# NCCL_PROTO=Simple / LL128      -15% single-stream decode
# NCCL_MIN_NCHANNELS=8|32, NCCL_ALGO=Ring, HIP_FORCE_DEV_KERNARG=1, GPU_MAX_HW_QUEUES=1|8, HSA_ENABLE_SDMA=0   within noise
# host C6 off / C1E+C6 off       within noise

# Host side, always: the CPU package RAPL-capped at 150 W while GPU jobs run (the 1228 W DC envelope; see README s.1).

# ADOPTED 2026-09-08 (fork knobs + gate sweeps, reports/2026-09-08-next-steps-measurements.md) for builds that carry the
# GGML_TP_AR_MAX_NE knob (fusion tree 25e1d46 and later; NOT the 2026-09-07 production binary, where the fork's default gate
# costs 8% at 8-16 slots): the fork's peer-write custom allreduce + whole-token graph, gated to messages of at most 4 decode
# rows (n_embd 5120). Single stream 49.4 -> 56.6 tok/s (+14.5%), 2 slots +19%, 4 slots +1-3%, 8+ unchanged, perplexity identical.
# Never GGML_TP_AR_NO_GATE=1 (prefill -36%). HSA_FORCE_FINE_GRAIN_PCIE alone is free.
GGML_ENABLE_CUSTOM_AR=1
HSA_FORCE_FINE_GRAIN_PCIE=1
GGML_TP_AR_MAX_NE=20481
