# qwen38-27b-mxxm-fh-server  2026-09-07T18:40:59+00:00  llama-server tp4 -np 16 -c 524288 -b 2048 -ub 2048 -fa on; server-bench conc 4,8,12,16, 1300-token prompts, 256 gen; production build then stock

## llama.cpp-mxxm-fh
# qwen38-27b-mxxm-fh-server-llama.cpp-mxxm-fh  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T18:41:46
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 31.4 | 65.1 | 396 | 15.3 | 28.2 | 4.96 | 15.7 |
| 8 | 16 | 54.2 | 75.5 | 459 | 17.7 | 14.5 | 5.25 | 27.0 |
| 12 | 24 | 74.0 | 83.1 | 505 | 19.5 | 10.1 | 5.65 | 36.8 |
| 16 | 32 | 99.7 | 82.1 | 499 | 19.3 | 7.3 | 5.63 | 49.7 |

## llama.cpp
# qwen38-27b-mxxm-fh-server-llama.cpp  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T18:46:53
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 35.4 | 57.9 | 352 | 13.6 | 27.0 | 6.20 | 17.7 |
| 8 | 16 | 63.0 | 65.1 | 395 | 15.2 | 13.0 | 6.66 | 31.3 |
| 12 | 24 | 104.5 | 58.8 | 357 | 13.8 | 6.7 | 7.29 | 52.0 |
| 16 | 32 | 128.7 | 63.6 | 387 | 14.9 | 5.5 | 7.38 | 64.1 |
# done 2026-09-07T18:52:29+00:00
