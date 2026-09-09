# post27 2026-09-08T19:38:44+00:00: branch r3 (-funsafe-math-optimizations restored), --no-repack unless noted; production for reference

## batched 8/12/16 at 2K, interleaved x2
| round | build | 8 | 12 | 16 |
|---|---|---:|---:|---:|
| 1 | r3 | 151.69 | 189.44 | 196.00 | 
| 1 | prod | 174.73 | 197.27 | 202.82 | 
| 2 | r3 | 154.00 | 194.78 | 195.88 | 
| 2 | prod | 174.60 | 197.08 | 202.85 | 

## server level team16, conc 8,16

### r3
# post27-r3  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-08T19:50:40
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 55.7 | 73.6 | 447 | 17.2 | 14.7 | 6.55 | 27.7 |
| 16 | 32 | 103.0 | 79.5 | 483 | 18.6 | 7.1 | 6.05 | 51.3 |

### prod
# post27-prod  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-08T19:54:00
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 53.1 | 77.2 | 469 | 18.1 | 15.4 | 6.16 | 26.4 |
| 16 | 32 | 100.2 | 81.8 | 497 | 19.2 | 7.2 | 5.77 | 49.9 |

## batched 8/12/16 with the repack on (r3)
 1 : 36.26 2 : 56.46 4 : 130.25 8 : 162.91 12 : 140.65 16 : 169.41 24 : 207.07 32 : 259.33 
# done 2026-09-08T20:00:53+00:00
