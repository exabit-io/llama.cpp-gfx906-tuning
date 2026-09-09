# qwen38-27b-nq-test  2026-09-08T09:22:52+00:00  norm+q8_1 fusion build vs production build
A. greedy completion DIFFERS (see qwen38-27b-nq-test-*-greedy.txt; first differing line: 3c3 < The tensor split also keeps the two dies' memory controllers on the same side of the problem. Each die owns its own HBM stack and its own slice of the model weights, and the only thing that cros)
B. perplexity nq build tp4 -c 16384 --chunks 6: Final estimate: PPL = 5.6083 +/- 0.06220 (reference 5.5969 +/- 0.062)

## C. llama-bench -p 2048 -n 128 -r 3, tp4 and rocm0: nq (both fusions), prod, nq with the q8 fusion off, with the add-norm fusion off, with the GDN producer fusion off, nq again
| build | device | test | t/s |
|---|---|---|---:|
| nq | 1 | ROCm0/ROCm1/ROCm2/ROCm3 | pp2048 |
| nq | 1 | ROCm0/ROCm1/ROCm2/ROCm3 | tg128 |
| nq | 1 | ROCm0 | pp2048 |
| nq | 1 | ROCm0 | tg128 |
| prod | 1 | ROCm0/ROCm1/ROCm2/ROCm3 | pp2048 |
| prod | 1 | ROCm0/ROCm1/ROCm2/ROCm3 | tg128 |
| prod | 1 | ROCm0 | pp2048 |
| prod | 1 | ROCm0 | tg128 |
| nq-noq8 | 1 | ROCm0/ROCm1/ROCm2/ROCm3 | pp2048 |
| nq-noq8 | 1 | ROCm0/ROCm1/ROCm2/ROCm3 | tg128 |
| nq-noq8 | 1 | ROCm0 | pp2048 |
| nq-noq8 | 1 | ROCm0 | tg128 |
| nq-noaddnorm | 1 | ROCm0/ROCm1/ROCm2/ROCm3 | pp2048 |
| nq-noaddnorm | 1 | ROCm0/ROCm1/ROCm2/ROCm3 | tg128 |
| nq-noaddnorm | 1 | ROCm0 | pp2048 |
| nq-noaddnorm | 1 | ROCm0 | tg128 |
| nq-nogdn | 1 | ROCm0/ROCm1/ROCm2/ROCm3 | pp2048 |
| nq-nogdn | 1 | ROCm0/ROCm1/ROCm2/ROCm3 | tg128 |
| nq-nogdn | 1 | ROCm0 | pp2048 |
| nq-nogdn | 1 | ROCm0 | tg128 |
| nq-repeat | 1 | ROCm0/ROCm1/ROCm2/ROCm3 | pp2048 |
| nq-repeat | 1 | ROCm0/ROCm1/ROCm2/ROCm3 | tg128 |
| nq-repeat | 1 | ROCm0 | pp2048 |
| nq-repeat | 1 | ROCm0 | tg128 |

## D. batched-bench tp4 -npp 2048 -ntg 128 -npl 8,16
### llama.cpp-mxxm-fh-nq
|  2048 |    128 |    8 |  17408 |   15.630 |  1048.27 |    5.837 |   175.42 |   21.467 |   810.92 |
|  2048 |    128 |   16 |  34816 |   30.682 |  1067.99 |   10.021 |   204.37 |   40.703 |   855.37 |
### llama.cpp-mxxm-fh
|  2048 |    128 |    8 |  17408 |   14.404 |  1137.44 |    5.854 |   174.92 |   20.258 |   859.30 |
|  2048 |    128 |   16 |  34816 |   28.605 |  1145.54 |   10.017 |   204.46 |   38.622 |   901.47 |

## E. kernel trace, nq build, tp4 single stream (M1 method; production: 1866 kernels per token per die, 257 quantize_q8_1 + 176 k_bin_bcast adds, busy 20.4 ms)
```
== Agent 1: 65 tokens (delimiter void mul_mat_vec_q<(ggml_type)8, 1, false, false>(void const... grid 3973120), steady window 55 tokens, median per token:
  kernels 1466   span 20.68 ms   busy(union) 19.34 ms   idle-in-span 1.33 ms
  mmvq: 11.11 ms/433  mmq: 0.00 ms/0  fa: 0.41 ms/32  rccl: 3.41 ms/128  norm: 1.66 ms/209  cpy: 0.28 ms/70  other: 2.48 ms/594  rt: 0.00 ms/0
  weight streaming: 6.3 GiB in 11.11 ms of mmvq = 609 GB/s achieved
  unprofiled token 21.50 ms - busy 19.34 ms = 2.16 ms true idle per token (10%)
     127    1339.4 void rms_norm_f32<1024, true, false, true, true>(float const*, float*,
     129     503.5 void quantize_q8_1<32>(float const*, void*, long, long, long, long, lo
      80     303.0 void rms_norm_f32<256, true, false, false, false>(float const*, float*
      48     189.1 void k_bin_bcast<&(op_add(float, float)), float, float, float, float c
      48     179.2 void unary_gated_op_kernel<&(op_softplus(float)), float>(float const*,
      16      63.7 void unary_gated_op_kernel<&(op_sigmoid(float)), float>(float const*, 
       1      11.0 void rms_norm_f32<1024, true, false, true, false>(float const*, float*
       1       7.7 void rms_norm_f32<1024, true, false, false, true>(float const*, float*
  kernels 1466   span 20.65 ms   busy(union) 19.30 ms   idle-in-span 1.35 ms
  mmvq: 11.10 ms/433  mmq: 0.00 ms/0  fa: 0.41 ms/32  rccl: 3.37 ms/128  norm: 1.66 ms/209  cpy: 0.28 ms/70  other: 2.49 ms/594  rt: 0.00 ms/0
  weight streaming: 6.3 GiB in 11.10 ms of mmvq = 610 GB/s achieved
```
# done 2026-09-08T09:38:32+00:00
