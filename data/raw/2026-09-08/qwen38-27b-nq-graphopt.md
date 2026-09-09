# qwen38-27b-nq-graphopt  2026-09-08T11:37:32+00:00  multi-stream graph optimisation on the split (fusion build)
| variant | env | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |
|---|---|---:|---:|---:|---:|
| off | BENCH=off | 1130.29 ± 1.23 | 49.46 ± 1.38 | 327.96 ± 0.09 | 20.48 ± 0.04 | 
| opt1 | GGML_CUDA_GRAPH_OPT=1 | 1131.81 ± 1.54 | 49.53 ± 1.39 | 328.31 ± 0.35 | 20.41 ± 0.04 | 
| opt2 | GGML_CUDA_GRAPH_OPT=2 | 1130.51 ± 1.31 | 49.43 ± 1.39 | 328.12 ± 0.20 | 20.48 ± 0.04 | 
| ar | GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481 | 1130.58 ± 0.91 | 56.49 ± 1.62 | 328.00 ± 0.09 | 20.35 ± 0.04 | 
| ar-opt2 | GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481 GGML_CUDA_GRAPH_OPT=2 | 1130.29 ± 1.35 | 56.44 ± 1.64 | 328.23 ± 0.06 | 20.49 ± 0.04 | 
| off-rep | BENCH=off2 | 1129.58 ± 0.87 | 49.56 ± 1.33 | 328.21 ± 0.11 | 20.57 ± 0.04 | 

Perplexity with GGML_CUDA_GRAPH_OPT=2 on tp4: Final estimate: PPL = 5.5969 +/- 0.06197 (reference 5.5969)
# done 2026-09-08T11:50:32+00:00
