# qwen38-27b-server-final  2026-09-08T14:09:25+00:00  server level, 1300-token prompts / 256 generated

## team16: /opt/llama.cpp-prod -np 16 -c 524288, conc 4,8,12,16
# qwen38-27b-server-final-team16  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-08T14:10:08
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 30.4 | 67.4 | 410 | 15.8 | 30.0 | 4.98 | 15.2 |
| 8 | 16 | 53.8 | 76.1 | 463 | 17.8 | 15.0 | 5.40 | 26.8 |
| 12 | 24 | 74.0 | 83.0 | 505 | 19.5 | 10.1 | 5.66 | 36.8 |
| 16 | 32 | 98.3 | 83.4 | 507 | 19.5 | 7.4 | 5.77 | 49.0 |

## busy32: /opt/llama.cpp-prod -np 32 -c 1048576, conc 16,32
# qwen38-27b-server-final-busy32  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-08T14:15:16
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 96.4 | 85.0 | 517 | 19.9 | 7.6 | 6.50 | 48.0 |
| 32 | 64 | 204.9 | 80.0 | 486 | 18.7 | 3.3 | 6.37 | 101.9 |

## pairs8: two tp2 servers -np 8 -c 262144, conc 8,16 total
# qwen38-27b-server-final-pairs8  2 servers  prompt~1300 tok  gen=256 tok (ignore_eos)  2026-09-08T14:21:07
| conc (total) | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s | per-server agg gen t/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 16 | 43.4 | 94.4 | 574 | 22.1 | 19.7 | 6.86 | 21.5 | 47.2 / 47.2 |
| 16 | 32 | 80.4 | 101.9 | 619 | 23.9 | 10.3 | 8.89 | 39.6 | 50.9 / 50.9 |

## prod0907-np32: /opt/llama.cpp-mxxm-fh -np 32 -c 1048576, conc 16,32
# qwen38-27b-server-final-prod0907-np32  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-08T14:24:08
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 98.3 | 83.4 | 507 | 19.5 | 7.5 | 6.65 | 48.9 |
| 32 | 64 | 206.6 | 79.3 | 482 | 18.6 | 3.3 | 6.41 | 102.8 |
# done 2026-09-08T14:29:17+00:00
