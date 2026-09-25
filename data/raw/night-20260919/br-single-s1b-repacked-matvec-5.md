
llama_batched_bench: n_kv_max = 262144, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -2, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|260864 |   1024 |    1 | 261888 |  655.121 |   398.19 |   33.173 |    30.87 |  688.293 |   380.49 |

