# qwen38-27b-q8_0-knobs-server  2026-09-07T10:47:30+00:00  tp4 -np 8 -c 262144; server-bench conc 1,8; base vs NCCL_PROTO=LL, interleaved twice
## base-1 (OPT=base)
# qwen38-27b-q8_0-knobs-server-base-1  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T10:47:58
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 60.9 | 33.6 | 204 | 7.9 | 46.2 | 1.81 | 7.6 |
| 8 | 16 | 62.0 | 66.1 | 402 | 15.5 | 12.8 | 6.39 | 30.9 |
## ll-1 (NCCL_PROTO=LL)
# qwen38-27b-q8_0-knobs-server-ll-1  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T10:50:33
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 61.7 | 33.2 | 202 | 7.8 | 46.1 | 1.87 | 7.7 |
| 8 | 16 | 62.2 | 65.9 | 400 | 15.4 | 13.0 | 6.59 | 31.0 |
## base-2 (OPT=base)
# qwen38-27b-q8_0-knobs-server-base-2  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T10:53:08
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 61.0 | 33.6 | 204 | 7.9 | 46.1 | 1.81 | 7.6 |
| 8 | 16 | 62.0 | 66.1 | 402 | 15.5 | 12.8 | 6.38 | 30.9 |
## ll-2 (NCCL_PROTO=LL)
# qwen38-27b-q8_0-knobs-server-ll-2  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T10:55:43
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 61.5 | 33.3 | 202 | 7.8 | 46.2 | 1.87 | 7.7 |
| 8 | 16 | 62.2 | 65.8 | 400 | 15.4 | 13.0 | 6.59 | 31.0 |

## GGML_CUDA_GRAPH_OPT=1 vs default, rocm0 alone, llama-bench -p 2048 -n 256 -d 0 -r 3 (interleaved twice)
| variant | pp2048 | tg256 |
|---|---:|---:|
| base (OPT=base) | 233.00 | 19.63 |
| graphopt (GGML_CUDA_GRAPH_OPT=1) | 233.06 | 19.64 |
| base2 (OPT=base) | 232.98 | 19.62 |
| graphopt2 (GGML_CUDA_GRAPH_OPT=1) | 233.15 | 19.62 |
# done 2026-09-07T11:03:14+00:00
