
llama_batched_bench: n_kv_max = 36864, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|  2048 |    128 |    8 |  17408 |   16.358 |  1001.61 |    6.394 |   160.16 |   22.751 |   765.15 |
|  2048 |    128 |   16 |  34816 |   28.647 |  1143.84 |   10.903 |   187.84 |   39.550 |   880.30 |

