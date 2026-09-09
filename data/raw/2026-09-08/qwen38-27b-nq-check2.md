# qwen38-27b-nq-check2  2026-09-08T09:53:27+00:00  fusion build follow-up

## A. greedy 200 tokens vs production (first differing character position; -1 = identical)
| variant | env | differs at char |
|---|---|---:|
| nq | BENCH=nq | 556 |
| noq8 | GGML_CUDA_NORM_Q8=0 | 556 |
| noaddnorm | GGML_CUDA_ADD_NORM=0 | 556 |
| nogdn | GGML_CUDA_GDN_PREFUSE=0 | -1 |
| alloff | GGML_CUDA_NORM_Q8=0 GGML_CUDA_ADD_NORM=0 GGML_CUDA_GDN_PREFUSE=0 | -1 |
| q8only | GGML_CUDA_ADD_NORM=0 GGML_CUDA_GDN_PREFUSE=0 | -1 |
| addnormonly | GGML_CUDA_NORM_Q8=0 GGML_CUDA_GDN_PREFUSE=0 | -1 |
| gdnonly | GGML_CUDA_NORM_Q8=0 GGML_CUDA_ADD_NORM=0 | 556 |

## B. perplexity tp4 -c 16384 --chunks 6 (reference 5.5969 +/- 0.062 on production)
| variant | PPL |
|---|---|
| alloff | 5.5969 +/- 0.06197 |
| q8only | 5.5969 +/- 0.06197 |
| addnormonly | 5.5969 +/- 0.06197 |
| gdnonly | 5.6083 +/- 0.06220 |

## C. llama-bench -p 2048 -n 128 -r 3 (raw md kept as qwen38-27b-nq-check2-<variant>-lb.md)
| variant | env | tp4 pp2048 | tp4 tg128 | die0 pp2048 | die0 tg128 |
|---|---|---:|---:|---:|---:|
| prod | BENCH=prod | 1127.56 ± 1.26 | 48.17 ± 1.34 | 321.71 ± 0.30 | 20.32 ± 0.04 | 
| nq | BENCH=nq | 1051.97 ± 0.60 | 49.51 ± 1.37 | 310.93 ± 0.14 | 20.45 ± 0.04 | 
| alloff | GGML_CUDA_NORM_Q8=0 GGML_CUDA_ADD_NORM=0 GGML_CUDA_GDN_PREFUSE=0 | 1127.68 ± 1.29 | 47.97 ± 1.30 | 327.85 ± 0.16 | 20.62 ± 0.04 | 
| noq8 | GGML_CUDA_NORM_Q8=0 | 1053.96 ± 0.91 | 49.81 ± 1.42 | 310.90 ± 0.03 | 20.61 ± 0.04 | 
| noaddnorm | GGML_CUDA_ADD_NORM=0 | 1048.54 ± 1.36 | 48.70 ± 1.37 | 310.53 ± 0.10 | 20.62 ± 0.04 | 
| nogdn | GGML_CUDA_GDN_PREFUSE=0 | 1135.21 ± 0.70 | 48.61 ± 1.31 | 328.50 ± 0.40 | 20.34 ± 0.04 | 
| prod-repeat | BENCH=prod2 | 1129.55 ± 1.26 | 48.05 ± 1.39 | 322.24 ± 0.04 | 20.30 ± 0.04 | 
| nq-repeat | BENCH=nq2 | 1053.42 ± 0.93 | 49.44 ± 1.34 | 310.95 ± 0.11 | 20.45 ± 0.04 | 

## D. batched-bench tp4 -npp 2048 -ntg 128 -npl 8,16, fusion build with all fusions off (first test: nq 1048/1068 pp vs prod 1137/1146)
|  2048 |    128 |    8 |  17408 |   14.816 |  1105.83 |    5.864 |   174.61 |   20.680 |   841.76 |
|  2048 |    128 |   16 |  34816 |   28.652 |  1143.64 |   10.039 |   204.00 |   38.692 |   899.83 |
# done 2026-09-08T10:19:34+00:00
