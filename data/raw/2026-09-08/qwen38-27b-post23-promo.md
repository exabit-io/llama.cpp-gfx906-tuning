# post23 promotion repeat 2026-09-08T18:01:04+00:00: branch r2 --no-repack vs production, interleaved x2

## batched-bench decode tok/s at 2K, tp4 -npl 8,12,16
| round | build | 8 | 12 | 16 |
|---|---|---:|---:|---:|
| 1 | r2 | 163.57 | 192.33 | 201.31 | 
| 1 | prod | 174.95 | 197.52 | 202.85 | 
| 2 | r2 | 153.94 | 195.78 | 192.53 | 
| 2 | prod | 174.81 | 197.51 | 202.93 | 

## server level, team16 (-np 16, 16 x 32K), 1300/256, conc 8,16

### round 1 r2
# post23-promo-r2-r1  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-08T18:09:17
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 55.5 | 73.8 | 449 | 17.3 | 14.6 | 6.41 | 27.6 |
| 16 | 32 | 103.1 | 79.5 | 483 | 18.6 | 7.1 | 6.02 | 51.3 |

### round 1 prod
# post23-promo-prod-r1  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-08T18:12:32
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 53.0 | 77.3 | 470 | 18.1 | 15.4 | 6.14 | 26.4 |
| 16 | 32 | 100.2 | 81.8 | 497 | 19.2 | 7.2 | 5.77 | 49.9 |

### round 2 r2
# post23-promo-r2-r2  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-08T18:15:41
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 54.9 | 74.6 | 454 | 17.5 | 14.6 | 6.25 | 27.3 |
| 16 | 32 | 102.7 | 79.7 | 485 | 18.7 | 7.1 | 5.99 | 51.2 |

### round 2 prod
# post23-promo-prod-r2  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-08T18:18:55
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 52.6 | 77.9 | 474 | 18.3 | 15.0 | 5.80 | 26.1 |
| 16 | 32 | 99.8 | 82.1 | 499 | 19.2 | 7.2 | 5.76 | 49.7 |
