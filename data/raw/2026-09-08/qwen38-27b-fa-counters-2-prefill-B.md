
llama_batched_bench: n_kv_max = 69632, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
| 16384 |      1 |    4 |  65540 |   94.671 |   692.25 |    0.401 |     9.98 |   95.072 |   689.37 |

