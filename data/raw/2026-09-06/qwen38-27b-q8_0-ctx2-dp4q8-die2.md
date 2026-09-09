# qwen38-27b-q8_0-ctx2 dp4-q8 die2  2026-09-06T19:54:31+00:00
# flags: --device rocm2 -sm none -fa on -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 -ntg 128 -npl 8 -npp 2048,8192 -c 67584 (4 instances concurrently)

llama_batched_bench: n_kv_max = 67584, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|  2048 |    128 |    8 |  17408 |   70.193 |   233.41 |   20.320 |    50.39 |   90.514 |   192.32 |
|  8192 |    128 |    8 |  66560 |  289.471 |   226.40 |   22.266 |    45.99 |  311.737 |   213.51 |

# exit=0  2026-09-06T20:01:22+00:00
