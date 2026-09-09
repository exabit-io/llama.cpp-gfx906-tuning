# qwen38-27b-q8_0-ctx server b4-f16: llama-server --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 4 -cb -c 1048576 -b 2048 -ub 2048 --cache-ram 0 ; gen 256, 4 simultaneous requests  2026-09-06T17:29:18+00:00
| prompt tok | conc | reqs | wall s | agg gen t/s | per-req gen t/s | per-req prefill t/s | TTFT mean / max s | req wall mean / max s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 130816 | 4 | 4 | 1055.3 | 1.0 | 6.1 | 403 | 400.2 / 784.6 | 1051.8 / 1055.3 |
# level 130816 exit=0 2026-09-06T17:47:01+00:00
| 261888 | 4 | 4 | 2923.3 | 0.4 | 6.2 | 294 | 1103.3 / 2182.9 | 2734.6 / 2923.3 |
# level 261888 exit=0 2026-09-06T18:35:54+00:00
