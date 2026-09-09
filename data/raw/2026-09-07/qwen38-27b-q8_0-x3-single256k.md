# qwen38-27b-q8_0-x3: one 256K request alone, per-slot server (-np 4 -c 1048576) vs the shared pool of extras2  2026-09-07T01:49:30+00:00
| server | prompt | conc | reqs | wall s | agg gen | per-req gen t/s | per-req prefill t/s | TTFT s | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| per-slot 4 x 256K | 261888 | 1 | 1 | 725.0 | 0.4 | 30.6 | 366 | 716.2 / 716.2 | 725.0 / 725.0 |
