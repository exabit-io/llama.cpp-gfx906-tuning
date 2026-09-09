# qwen38-27b-q8_0-x2 kvu-b8  2026-09-06T20:57:07+00:00
# flags: env OPT=kvu llama-batched-bench --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -b 2048 -ub 2048 --kv-unified -c 262144 -ntg 128 -npl 8 -npp 2048,8192,32512

llama_batched_bench: n_kv_max = 262144, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|  2048 |    128 |    8 |  17408 |   20.990 |   780.56 |    6.924 |   147.90 |   27.914 |   623.64 |
|  8192 |    128 |    8 |  66560 |  102.264 |   640.85 |    7.682 |   133.31 |  109.946 |   605.39 |
| 32512 |    128 |    8 | 261120 |  701.664 |   370.68 |   11.887 |    86.14 |  713.551 |   365.94 |

# exit=0  2026-09-06T21:11:39+00:00
