
llama_batched_bench: n_kv_max = 36864, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|  2048 |    128 |    8 |  17408 |   15.108 |  1084.49 |    5.877 |   174.25 |   20.984 |   829.57 |
|  2048 |    128 |   16 |  34816 |   28.662 |  1143.24 |   10.032 |   204.14 |   38.695 |   899.76 |

