# qwen38-27b-q8_0-x2 kvu-b4  2026-09-06T21:11:39+00:00
# flags: env OPT=kvu llama-batched-bench --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -b 2048 -ub 2048 --kv-unified -c 262144 -ntg 128 -npl 4 -npp 32768,65408

llama_batched_bench: n_kv_max = 262144, n_batch = 2048, n_ubatch = 2048, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 28, n_threads_batch = 28

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
| 32768 |    128 |    4 | 131584 |  255.341 |   513.32 |    5.837 |    87.72 |  261.177 |   503.81 |
| 65408 |    128 |    4 | 262144 |  708.578 |   369.24 |    6.992 |    73.22 |  715.571 |   366.34 |

# exit=0  2026-09-06T21:28:14+00:00
