# qwen38-27b-q8_0-ctx server b8-f16: llama-server --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 8 -cb -c 1048576 -b 2048 -ub 2048 --cache-ram 0 ; gen 256, 8 simultaneous requests  2026-09-06T16:45:48+00:00
| prompt tok | conc | reqs | wall s | agg gen t/s | per-req gen t/s | per-req prefill t/s | TTFT mean / max s | req wall mean / max s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 8 | 8 | 59.0 | 34.7 | 10.7 | 337 | 15.5 / 41.2 | 58.3 / 59.0 |
# level 4096 exit=0 2026-09-06T16:46:54+00:00
| 32768 | 8 | 8 | 390.4 | 5.2 | 3.8 | 536 | 90.0 / 335.3 | 387.8 / 390.4 |
# level 32768 exit=0 2026-09-06T16:53:31+00:00
| 130816 | 8 | 8 | 2111.9 | 1.0 | 3.1 | 425 | 470.8 / 1839.9 | 1907.7 / 2111.9 |
# level 130816 exit=0 2026-09-06T17:28:53+00:00
