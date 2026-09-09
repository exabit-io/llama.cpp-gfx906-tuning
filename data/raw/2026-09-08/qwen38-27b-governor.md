# qwen38-27b-governor  2026-09-08T14:29:31+00:00  16-client serving (tp4 -np 16, production build) beside an all-core host load at stepped RAPL caps

## A. serving cost of host load at the 150 W cap: server-bench conc 16, host idle vs all-core load
# qwen38-27b-governor-idle  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-08T14:30:14
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 99.6 | 82.3 | 500 | 19.3 | 7.4 | 6.50 | 49.6 |
# qwen38-27b-governor-load150  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-08T14:32:12
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 113.7 | 72.1 | 438 | 16.9 | 6.3 | 6.67 | 56.5 |

## B. DC total (SMC PZ0G) vs host RAPL cap with the 16 clients running; stop at 1180 W
| host cap W | DC W (max over 40 s) | die W mean |
|---:|---:|---:|
| 150 | 1133 | 107 |
| 175 | 1172 | 147 |
| 200 | 1206 | 153 |
# done 2026-09-08T14:35:46+00:00
