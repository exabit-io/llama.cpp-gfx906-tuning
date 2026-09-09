# qwen38-27b-q8_0-x2 K: llama-server --device rocm0,rocm1,rocm2,rocm3 -sm tensor -np 8 -c 262144 --kv-unified (one 256K pool shared by 8 slots), f16, gen 256  2026-09-06T21:28:14+00:00
| server | wave | wall s | agg gen t/s | per-req gen t/s | per-req prefill t/s | TTFT mean / max s | req wall mean / max s |
|---|---|---:|---:|---:|---:|---:|---:|
| unified 8 x 256K pool | 32512 | 8 | 8 | 744.3 | 2.8 | 2.2 | 318 | 165.8 / 609.9 | 740.9 / 744.3 |
| unified 8 x 256K pool | 261888 | 1 | 1 | 1401.9 | 0.2 | 30.4 | 188 | 1393.1 / 1393.1 | 1401.9 / 1401.9 |
| unified 8 x 256K pool | 4096 | 8 | 8 | 66.2 | 30.9 | 9.5 | 285 | 15.2 / 20.0 | 65.4 / 66.2 |

| server | mix | wall s | long TTFT s | long gen t/s | long wall s | short TTFT mean / max s | short gen t/s | short wall mean / max s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| unified 8 x 256K pool | 1 × 130,816 + 7 × 4,096 | 384 | 308 | 3.6 | 382 | 108.6 / 331.8 | 7.5 | 383 / 384 |
| per-slot 8 x 128K | 1 × 130,816 + 7 × 4,096 | 322 | 264 | 12.9 | 322 | 17.4 / 43.5 | 0.9 | 320 / 320 |
# done 2026-09-06T22:18:07+00:00
