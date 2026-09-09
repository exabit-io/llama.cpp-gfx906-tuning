# qwen38-27b-q8_0-server-layer4  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-03T11:27:00
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 152.7 | 13.4 | 82 | 3.1 | 19.6 | 5.89 | 19.1 |
| 2 | 8 | 99.7 | 20.6 | 125 | 4.8 | 17.6 | 10.21 | 24.9 |
| 4 | 8 | 78.1 | 26.2 | 159 | 6.1 | 11.8 | 13.77 | 39.0 |
| 8 | 16 | 159.0 | 25.8 | 157 | 6.0 | 4.9 | 16.84 | 79.4 |
| 16 | 32 | 301.7 | 27.2 | 165 | 6.4 | 2.4 | 18.95 | 150.5 |
# client exit=1 2026-09-03T11:52:57+00:00

# conc 32 retry after GPU page fault on 1b:00.0 at 11:46 UTC (first attempt), fresh server 2026-09-03T12:09:06+00:00
# qwen38-27b-q8_0-server-layer4-c32retry  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-03T12:09:25
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 64 | 540.4 | 30.3 | 184 | 7.1 | 1.2 | 16.99 | 269.1 |
# client exit=0 2026-09-03T12:18:26+00:00
