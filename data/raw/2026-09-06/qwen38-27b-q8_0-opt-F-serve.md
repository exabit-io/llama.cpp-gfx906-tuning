# qwen38-27b-q8_0-opt F: server tensor 4 dies -np 8 -c 32768, server-bench.py conc 1,4,8 (1300-token prompts, 256 gen); base vs best A variant 'topo-fixed-16ch' (NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 -sm tensor)  2026-09-06T19:33:04+00:00
## base (OPT=base)
# qwen38-27b-q8_0-opt-F-base  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-06T19:33:42
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 63.1 | 32.5 | 197 | 7.6 | 44.7 | 1.83 | 7.9 |
| 4 | 8 | 34.2 | 59.8 | 364 | 14.0 | 26.4 | 5.77 | 17.1 |
| 8 | 16 | 63.1 | 64.9 | 395 | 15.2 | 12.7 | 6.54 | 31.5 |
## best (NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 -sm tensor)
server failed
# done 2026-09-06T19:36:28+00:00
## best-rerun (NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16) 2026-09-06T20:53:45+00:00
# qwen38-27b-q8_0-opt-F-best  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-06T20:53:53
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 61.0 | 33.6 | 204 | 7.9 | 46.1 | 1.81 | 7.6 |
| 4 | 8 | 34.6 | 59.1 | 359 | 13.9 | 25.4 | 5.36 | 17.2 |
| 8 | 16 | 61.7 | 66.3 | 403 | 15.5 | 13.0 | 6.41 | 30.8 |
