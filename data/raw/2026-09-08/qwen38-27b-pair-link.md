# qwen38-27b-pair-link 2026-09-08T20:46:42+00:00: tp2 on three pairs of the ring as cabled (production build, gfx906.env)
| pair | link | pp2048 | tg128 | batched 8 slots decode | batched 8 prefill |
|---|---|---:|---:|---:|---:|
| on-card | one direct XGMI link (0b-0e) | 607.40 ± 0.51 | 37.84 ± 0.11 |  113.27 | 609.60 |
| bridge | one direct XGMI link (0b-1b) | 620.46 ± 0.12 | 38.26 ± 0.11 |  115.47 | 614.85 |
| diagonal | no direct link (0b-1e: two hops or PCIe) | 591.97 ± 0.34 | 38.29 ± 0.11 |  113.63 | 582.76 |
# done 2026-09-08T20:51:31+00:00
