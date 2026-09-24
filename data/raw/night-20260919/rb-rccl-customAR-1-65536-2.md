
llama_batched_bench: n_kv_max = 66816, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -2, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
| 65536 |   1024 |    1 |  66560 |   87.716 |   747.14 |   29.597 |    34.60 |  117.313 |   567.37 |

