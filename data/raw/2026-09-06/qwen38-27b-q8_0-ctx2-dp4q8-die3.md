# qwen38-27b-q8_0-ctx2 dp4-q8 die3  2026-09-06T19:54:31+00:00
# flags: --device rocm3 -sm none -fa on -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 -ntg 128 -npl 8 -npp 2048,8192 -c 67584 (4 instances concurrently)

llama_batched_bench: n_kv_max = 67584, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|  2048 |    128 |    8 |  17408 |   70.055 |   233.87 |   20.121 |    50.89 |   90.177 |   193.04 |
|  8192 |    128 |    8 |  66560 |  288.620 |   227.07 |   22.159 |    46.21 |  310.779 |   214.17 |

# exit=0  2026-09-06T20:01:21+00:00
