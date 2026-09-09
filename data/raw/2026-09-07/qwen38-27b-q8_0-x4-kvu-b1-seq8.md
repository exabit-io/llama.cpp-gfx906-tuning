# qwen38-27b-q8_0-x4 kvu-b1-seq8  2026-09-07T02:28:11+00:00

llama_batched_bench: n_kv_max = 262144, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|262016 |    128 |    1 | 262144 |  716.696 |   365.59 |    4.182 |    30.61 |  720.878 |   363.65 |

# exit=0  2026-09-07T02:40:29+00:00
