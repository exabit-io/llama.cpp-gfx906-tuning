# qwen38-27b-q8_0-opt D: Qwen3.5-35B-A3B Q8_0 (qwen35moe)  2026-09-06T19:17:27+00:00
## llama-bench 2 and 4 dies, -d 0,32768
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35moe 35B.A3B Q8_0         |  34.36 GiB |    34.66 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |          pp2048 |       1955.27 ± 1.88 |
| qwen35moe 35B.A3B Q8_0         |  34.36 GiB |    34.66 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |           tg256 |         82.19 ± 1.16 |
| qwen35moe 35B.A3B Q8_0         |  34.36 GiB |    34.66 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  | pp2048 @ d32768 |      1269.29 ± 11.97 |
| qwen35moe 35B.A3B Q8_0         |  34.36 GiB |    34.66 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |  tg256 @ d32768 |         76.17 ± 0.96 |
| qwen35moe 35B.A3B Q8_0         |  34.36 GiB |    34.66 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |       2544.82 ± 1.72 |
| qwen35moe 35B.A3B Q8_0         |  34.36 GiB |    34.66 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         77.02 ± 2.78 |
| qwen35moe 35B.A3B Q8_0         |  34.36 GiB |    34.66 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 | pp2048 @ d32768 |      1484.94 ± 18.61 |
| qwen35moe 35B.A3B Q8_0         |  34.36 GiB |    34.66 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |  tg256 @ d32768 |         71.72 ± 1.57 |

build: 360e134 (1)
## batched-bench 4 dies, pp 2048, batch 1/4/8/16/32

llama_batched_bench: n_kv_max = 73728, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|  2048 |    128 |    1 |   2176 |    0.847 |  2418.41 |    1.735 |    73.77 |    2.582 |   842.77 |
|  2048 |    128 |    4 |   8704 |    3.224 |  2541.33 |    4.124 |   124.16 |    7.347 |  1184.65 |
|  2048 |    128 |    8 |  17408 |    6.430 |  2548.07 |    3.854 |   265.70 |   10.284 |  1692.74 |
|  2048 |    128 |   16 |  34816 |   12.878 |  2544.43 |    7.434 |   275.49 |   20.312 |  1714.04 |
|  2048 |    128 |   32 |  69632 |   25.684 |  2551.65 |   12.171 |   336.54 |   37.855 |  1839.45 |

## batched-bench 4 dies, batch 8, depth 8K / 32K

llama_batched_bench: n_kv_max = 264192, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|  8192 |    128 |    8 |  66560 |   28.505 |  2299.15 |    7.364 |   139.06 |   35.868 |  1855.68 |
| 32768 |    128 |    8 | 263168 |  135.972 |  1927.92 |    4.407 |   232.38 |  140.379 |  1874.70 |

# done 2026-09-06T19:25:34+00:00
