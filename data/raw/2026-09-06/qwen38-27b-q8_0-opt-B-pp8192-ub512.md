
llama_batched_bench: n_kv_max = 67584, n_batch = 2048, n_ubatch = 512, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|  8192 |    128 |    8 |  66560 |   95.161 |   688.69 |    7.707 |   132.87 |  102.867 |   647.05 |

