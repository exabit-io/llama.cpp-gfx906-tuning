
llama_batched_bench: n_kv_max = 135168, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
| 32768 |     16 |    4 | 131136 |  203.405 |   644.39 |    4.000 |    16.00 |  207.405 |   632.27 |

