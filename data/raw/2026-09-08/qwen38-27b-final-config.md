# qwen38-27b-final-config  2026-09-08T11:52:26+00:00  final = /opt/llama.cpp-mxxm-fh-nq with: GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481 GGML_CUDA_GDN_PREFUSE=7  vs production /opt/llama.cpp-mxxm-fh (libggml-hip: /opt/llama.cpp/lib/libggml-hip.so.0)

## A0. decode-path numerics vs production: greedy 200 tokens differ at char 556 (-1 = identical); KL divergence at 64-token batches over 8 x 2048 tokens: chunk PPL ln(PPL(Q)/PPL(base)) KL Divergence Δp RMS Same top p;Mean ln(PPL(Q)/PPL(base)) : -0.000000 ± 0.000000;Mean KLD: 0.000000 ± 0.000000;Maximum KLD: 0.000067;Same top p: 100.000 ± 0.000 %;

## A. llama-bench -p 2048 -n 128 -r 3
| build | env | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |
|---|---|---:|---:|---:|---:|
| prod | BENCH=prod | 1129.66 ± 1.25 | 48.32 ± 1.31 | 322.04 ± 0.19 | 20.66 ± 0.05 | 
| final | GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481 GGML_CUDA_GDN_PREFUSE=7 | 1129.89 ± 1.33 | 58.08 ± 1.82 | 331.60 ± 0.12 | 21.76 ± 0.05 | 
| final-noar | GGML_CUDA_GDN_PREFUSE=7 | 1129.24 ± 1.11 | 50.57 ± 1.47 | 331.57 ± 0.27 | 21.73 ± 0.05 | 
| prod-repeat | BENCH=prod2 | 1130.75 ± 1.36 | 47.91 ± 1.33 | 321.97 ± 0.23 | 20.17 ± 0.04 | 
| final-repeat | GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481 GGML_CUDA_GDN_PREFUSE=7 | 1129.71 ± 1.41 | 57.78 ± 1.69 | 331.54 ± 0.09 | 21.71 ± 0.05 | 

## B. batched-bench tp4 -npp 2048 -ntg 128 -npl 1,2,4,8,16,32 (decode tok/s)
| build | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| prod | 47.71 | 76.61 | 122.95 | 174.74 | 203.90 | 212.66 | 
| final | 41.96 | 90.77 | 126.07 | 174.93 | 202.73 | 214.58 | 

## C. single user with MTP draft 3: llama-server -np 1, mtp-depth-client conc 1, 300 generated, two waves (wave 2 = decode only)
| build | depth | wall s | per-req gen t/s (wave 2) | accepted / drafted |
|---|---:|---:|---:|---|
| prod | 2048 | 4 | 75.3 | 201 / 291 = 0.69 |
| final | 2048 | 4 | 78.5 | 200 / 294 = 0.68 |
| prod | 32768 | 5 | 67.9 | 197 / 303 = 0.65 |
| final | 32768 | 5 | 73.9 | 201 / 294 = 0.68 |

## D. perplexity of the final configuration: Final estimate: PPL = 5.5969 +/- 0.06197 (production 5.5969 +/- 0.062)
# done 2026-09-08T12:22:09+00:00
