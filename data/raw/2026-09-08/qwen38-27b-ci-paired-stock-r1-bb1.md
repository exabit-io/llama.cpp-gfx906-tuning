
llama_batched_bench: n_kv_max = 6144, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   512 |    128 |    4 |   2560 |    8.471 |   241.78 |   11.085 |    46.19 |   19.555 |   130.91 |
|   512 |    128 |    8 |   5120 |   16.933 |   241.89 |   19.427 |    52.71 |   36.360 |   140.81 |

