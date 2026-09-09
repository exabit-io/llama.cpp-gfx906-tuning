# qwen38-27b-post22d 2026-09-08T17:01:48+00:00: gfx906 branch (r2) at the server level, team16 profile (-np 16, 16 x 32K), 1300/256 requests; production read 67.4 / 76.1 / 83.0 / 83.4 at 4 / 8 / 12 / 16 clients

## repack on
# qwen38-27b-post22d-on  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-08T17:03:06
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 33.0 | 62.0 | 377 | 14.5 | 29.3 | 6.04 | 16.5 |
| 8 | 16 | 53.9 | 76.0 | 462 | 17.8 | 14.4 | 5.15 | 26.8 |
| 12 | 24 | 84.6 | 72.6 | 441 | 17.0 | 8.1 | 5.26 | 42.1 |
| 16 | 32 | 102.9 | 79.6 | 484 | 18.7 | 6.6 | 5.19 | 51.2 |

## repack off
# qwen38-27b-post22d-off  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-08T17:08:22
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 30.9 | 66.3 | 403 | 15.5 | 29.5 | 5.04 | 15.4 |
| 8 | 16 | 55.7 | 73.5 | 447 | 17.2 | 14.2 | 5.55 | 27.7 |
| 12 | 24 | 76.9 | 79.9 | 486 | 18.7 | 9.8 | 5.94 | 38.3 |
| 16 | 32 | 100.9 | 81.2 | 494 | 19.0 | 7.1 | 5.88 | 50.2 |
# done 2026-09-08T17:12:51+00:00
