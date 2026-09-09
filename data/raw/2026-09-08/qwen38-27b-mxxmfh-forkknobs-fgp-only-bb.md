
llama_batched_bench: n_kv_max = 36864, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|  2048 |    128 |    8 |  17408 |   14.395 |  1138.15 |    5.858 |   174.82 |   20.253 |   859.53 |
|  2048 |    128 |   16 |  34816 |   28.612 |  1145.26 |   10.025 |   204.30 |   38.636 |   901.12 |

