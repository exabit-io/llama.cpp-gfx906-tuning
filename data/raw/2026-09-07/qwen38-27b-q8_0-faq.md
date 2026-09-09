# qwen38-27b-q8_0-faq  2026-09-07T09:58:30+00:00  GGML_CUDA_FA_ALL_QUANTS=ON build (/opt/llama.cpp-faq)

## q8k-q4v-depth: llama-batched-bench -ctk q8_0 -ctv q4_0 -c 278528 -npp 2048,32768 -ntg 128 -npl 8
|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|  2048 |    128 |    8 |  17408 |   19.363 |   846.14 |    6.701 |   152.82 |   26.064 |   667.90 |
| 32768 |    128 |    8 | 263168 |  357.445 |   733.38 |    9.532 |   107.42 |  366.978 |   717.12 |
(exit 0)  compute buffer:   KV: 

## q8k-q8v-depth: llama-batched-bench -ctk q8_0 -ctv q8_0 -c 278528 -npp 2048,32768 -ntg 128 -npl 8
|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|  2048 |    128 |    8 |  17408 |   19.377 |   845.54 |    6.684 |   153.21 |   26.061 |   667.98 |
| 32768 |    128 |    8 | 263168 |  357.548 |   733.17 |    9.213 |   111.15 |  366.761 |   717.55 |
(exit 0)  compute buffer:   KV: 

## f16-depth: llama-batched-bench -c 278528 -npp 2048,32768 -ntg 128 -npl 8
|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|  2048 |    128 |    8 |  17408 |   19.339 |   847.22 |    6.410 |   159.74 |   25.749 |   676.07 |
| 32768 |    128 |    8 | 263168 |  355.631 |   737.12 |    7.806 |   131.18 |  363.437 |   724.11 |
(exit 0)  compute buffer:   KV: 

## fit-8x256k-q8q8: llama-batched-bench -ctk q8_0 -ctv q8_0 -c 2097152 -npp 512 -ntg 32 -npl 8
(exit 134)  compute buffer:   KV: 

## fit-8x256k-q8q4: llama-batched-bench -ctk q8_0 -ctv q4_0 -c 2097152 -npp 512 -ntg 32 -npl 8
|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   512 |     32 |    8 |   4352 |    4.827 |   848.60 |    1.792 |   142.84 |    6.619 |   657.50 |
(exit 0)  compute buffer:   KV: 

## perplexity -ctk q8_0 -ctv q4_0, tp4, -c 16384 --chunks 6
0.16.241.064 I system_info: n_threads = 28 (n_threads_batch = 28) / 56 | ROCm : NO_VMM = 1 | FA_ALL_QUANTS = 1 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VNNI = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 | 

## perplexity -ctk q8_0 -ctv q8_0, tp4, -c 16384 --chunks 6
0.16.318.454 I system_info: n_threads = 28 (n_threads_batch = 28) / 56 | ROCm : NO_VMM = 1 | FA_ALL_QUANTS = 1 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VNNI = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 | 
# done 2026-09-07T10:26:02+00:00
