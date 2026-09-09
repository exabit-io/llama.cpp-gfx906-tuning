# qwen38-27b-mxxmfh-tp2x2  2026-09-08T11:03:32+00:00  two tp2 servers on the production build (8089 = rocm0,rocm1 / 8090 = rocm2,rocm3); libggml-hip: /opt/llama.cpp-mxxm-fh/lib/libggml-hip.so.0

## C1. two tp2 servers -np 8 -c 262144 each; dual-server-bench conc 8,16 total, 1300-token prompts, 256 gen
# qwen38-27b-mxxmfh-tp2x2-C1  2 servers  prompt~1300 tok  gen=256 tok (ignore_eos)  2026-09-08T11:04:08
| conc (total) | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s | per-server agg gen t/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 16 | 45.2 | 90.7 | 551 | 21.3 | 18.2 | 6.78 | 22.4 | 45.3 / 45.3 |
| 16 | 32 | 81.7 | 100.3 | 610 | 23.5 | 10.0 | 8.79 | 40.2 | 50.2 / 50.2 |

## C2. two tp2 servers -np 16 -c 131072 each; dual-server-bench conc 16,32 total, 1300-token prompts, 256 gen
# qwen38-27b-mxxmfh-tp2x2-C2  2 servers  prompt~1300 tok  gen=256 tok (ignore_eos)  2026-09-08T11:06:55
| conc (total) | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s | per-server agg gen t/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 32 | 80.6 | 101.7 | 618 | 23.8 | 9.7 | 7.92 | 39.7 | 50.8 / 50.8 |
| 32 | 64 | 169.0 | 97.0 | 589 | 22.7 | 4.3 | 9.35 | 82.8 | 48.5 / 48.5 |
# done 2026-09-08T11:11:07+00:00
