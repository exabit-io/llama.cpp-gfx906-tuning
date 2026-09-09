# post22 2026-09-08T15:50:32+00:00: the gfx906 branch build (r2, mul_mat_id fix) with and without the fork's weight repack; bisect perplexities

## llama-bench -p 2048 -n 128 -r 3 (tp4 + rocm0), gfx906.env
| build | repack | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |
|---|---|---:|---:|---:|---:|
| gfx906-master-r2 | off | 1137.04 ± 0.93 | 37.91 ± 16.39 | 338.75 ± 0.02 | 21.43 ± 0.10 | 
| gfx906-master-r2 | on | 1361.49 ± 2.15 | 53.71 ± 2.71 | 424.11 ± 0.03 | 21.90 ± 0.15 | 

## batched-bench decode tok/s at 2K, tp4, --no-repack (repack on: 1:25.8 2:39.8 4:127.2 8:174.4 12:141.4 16:169.2 24:219.7 32:258.6; production 1:57.9 8:174.6 12:197.1 16:202.8 32:212.7)
 1 : 31.34 2 : 59.32 4 : 129.37 8 : 158.48 12 : 196.52 16 : 201.61 24 : 194.07 32 : 212.82 

## bisect perplexities (16K, 6 chunks, tp4)
| build | commit | ppl 16K/6 |
|---|---|---|
| master 5d806aa25 pristine (/opt/llama.cpp-master) | 2026-09-08 | 5.6216 ± 0.06246 |
| gfx906 branch (fork b10912 + merge + series, /opt/llama.cpp-gfx906-master) | 2026-09-08 | 5.6173 ± 0.06235 |
| production (fork b10254 + series) | | 5.5969 |
| 0ba6499c3 | 2026-09-03 CUDA: Allow concurrent streams per split for multi-GPU (#28198) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| 3ad1ba733 | 2026-09-06 [Model] Support for Spark2_5ForCausalLM  implementation (#27868) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| 5a6caa05f | 2026-09-08 ggml : update ggml_prec specification (#26675) | Final estimate: PPL = 5.6216 +/- 0.06246 |
| 5fdfa6282 | 2026-09-06 models : fix GDN normalization from `max` to `rsqrt` (#28068) | Final estimate: PPL = 5.6216 +/- 0.06246 |
| 64a155d24 | 2026-09-04 sync : ggml (#28379) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| 73ab7599b | 2026-09-07 CUDA: branchless Q4_K/Q5_K unpack to speed up mmvq, L2 prefetch on DGX Spark (# | Final estimate: PPL = 5.6216 +/- 0.06246 |
| 8e93a9773 | 2026-09-02 CUDA + ggml: add sparse-fa for DSV4/GLM (#27970) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| 992cb503c | 2026-09-07 ggml: allow backend inputs to not create another split (#28387) | Final estimate: PPL = 5.6216 +/- 0.06246 |

## kernel trace, tp4 tg64 (llama-bench -p 0 -n 64 -r 2), repack on / off: top kernels and inter-kernel gaps
### repack on:          10.27 ± 1.35 

== Agent 1: 129 tokens (delimiter void mul_mat_vec_q<(ggml_type)8, 1, false, false, false>(voi... grid 3973120), steady window 119 tokens, median per token:
  token period (delimiter to delimiter) 21.02 ms; inter-token gap 0.40 ms
  kernels 1466   span 20.62 ms   busy(union) 20.01 ms   idle-in-span 0.58 ms
  mmvq: 10.78 ms/433  mmq: 0.00 ms/0  fa: 0.40 ms/32  rccl: 4.35 ms/128  norm: 1.68 ms/209  cpy: 0.28 ms/70  other: 2.52 ms/594  rt: 0.00 ms/0
  gaps per token: 0-10us: 5 (0.03 ms), 10-30us: 3 (0.05 ms), 30-100us: 1 (0.04 ms), 100-300us: 2 (0.25 ms), 300-1000000000us: 1 (11.38 ms)

== Agent 2: 129 tokens (delimiter void mul_mat_vec_q<(ggml_type)8, 1, false, false, false>(voi... grid 3973120), steady window 119 tokens, median per token:
  token period (delimiter to delimiter) 21.02 ms; inter-token gap 0.43 ms
  kernels 1466   span 20.59 ms   busy(union) 18.97 ms   idle-in-span 1.62 ms
  mmvq: 10.79 ms/433  mmq: 0.00 ms/0  fa: 0.38 ms/32  rccl: 3.29 ms/128  norm: 1.71 ms/209  cpy: 0.28 ms/70  other: 2.52 ms/594  rt: 0.00 ms/0
  gaps per token: 0-10us: 5 (0.03 ms), 10-30us: 3 (0.05 ms), 30-100us: 1 (0.06 ms), 100-300us: 1 (0.13 ms), 300-1000000000us: 2 (12.47 ms)

== Agent 3: 129 tokens (delimiter void mul_mat_vec_q<(ggml_type)8, 1, false, false, false>(voi... grid 3973120), steady window 119 tokens, median per token:
  token period (delimiter to delimiter) 21.02 ms; inter-token gap 0.47 ms
  kernels 1466   span 20.56 ms   busy(union) 18.07 ms   idle-in-span 2.46 ms
  mmvq: 10.76 ms/433  mmq: 0.00 ms/0  fa: 0.38 ms/32  rccl: 2.42 ms/128  norm: 1.70 ms/209  cpy: 0.28 ms/70  other: 2.52 ms/594  rt: 0.00 ms/0
  gaps per token: 0-10us: 5 (0.03 ms), 10-30us: 3 (0.05 ms), 30-100us: 1 (0.10 ms), 100-300us: 0 (0.05 ms), 300-1000000000us: 2 (22.74 ms)

== Agent 4: 129 tokens (delimiter void mul_mat_vec_q<(ggml_type)8, 1, false, false, false>(voi... grid 3973120), steady window 119 tokens, median per token:
  token period (delimiter to delimiter) 21.02 ms; inter-token gap 0.49 ms
  kernels 1466   span 20.53 ms   busy(union) 17.25 ms   idle-in-span 3.28 ms
  mmvq: 10.77 ms/433  mmq: 0.00 ms/0  fa: 0.40 ms/32  rccl: 1.50 ms/128  norm: 1.77 ms/209  cpy: 0.28 ms/70  other: 2.52 ms/594  rt: 0.00 ms/0
  gaps per token: 0-10us: 5 (0.03 ms), 10-30us: 3 (0.05 ms), 30-100us: 1 (0.08 ms), 100-300us: 0 (0.02 ms), 300-1000000000us: 2 (54.27 ms)

"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"void ggml_cuda_tp::k_broadcast_reduce<4>(ggml_cuda_tp::RankData*, ggml_cuda_tp::RankSignals, ggml_cuda_tp::Signal*, float const*, float*, int, long)"
"void mul_mat_vec_q8_0_repacked<16, 16, false, 64, false>(unsigned char const*, block_q8_1 const*, float*, unsigned int, unsigned int, int const*, int
"void mul_mat_vec_q8_0_repacked<16, 16, false, 64, true>(unsigned char const*, block_q8_1 const*, float*, unsigned int, unsigned int, int const*, int 
"void rms_norm_f32<1024, true, false, true, true>(float const*, float*, int, long, long, long, float, float const*, long, long, long, HIP_vector_type<
"void quantize_q8_1<32>(float const*, void*, long, long, long, long, long, unsigned int, HIP_vector_type<unsigned int, 3u>)",66564,261435394,3927.5793
"void mul_mat_vec_q<(ggml_type)8, 1, false, false, false>(void const*, void const*, int const*, ggml_cuda_mm_fusion_args_device, float*, unsigned int,
"void gated_delta_net_cuda<128, false, false>(float const*, float const*, float const*, float const*, float const*, float const*, float*, float*, long
### repack off:           9.19 ± 1.71 

== Agent 1: 129 tokens (delimiter void mul_mat_vec_q<(ggml_type)8, 1, false, false, false>(voi... grid 3973120), steady window 119 tokens, median per token:
  token period (delimiter to delimiter) 20.94 ms; inter-token gap 0.39 ms
  kernels 1466   span 20.55 ms   busy(union) 19.93 ms   idle-in-span 0.59 ms
  mmvq: 10.62 ms/433  mmq: 0.00 ms/0  fa: 0.40 ms/32  rccl: 4.37 ms/128  norm: 1.76 ms/209  cpy: 0.28 ms/70  other: 2.50 ms/594  rt: 0.00 ms/0
  gaps per token: 0-10us: 5 (0.03 ms), 10-30us: 3 (0.05 ms), 30-100us: 1 (0.03 ms), 100-300us: 2 (0.25 ms), 300-1000000000us: 1 (19.15 ms)

== Agent 2: 129 tokens (delimiter void mul_mat_vec_q<(ggml_type)8, 1, false, false, false>(voi... grid 3973120), steady window 119 tokens, median per token:
  token period (delimiter to delimiter) 20.94 ms; inter-token gap 0.44 ms
  kernels 1466   span 20.51 ms   busy(union) 18.83 ms   idle-in-span 1.68 ms
  mmvq: 10.60 ms/433  mmq: 0.00 ms/0  fa: 0.39 ms/32  rccl: 3.34 ms/128  norm: 1.70 ms/209  cpy: 0.28 ms/70  other: 2.52 ms/594  rt: 0.00 ms/0
  gaps per token: 0-10us: 5 (0.03 ms), 10-30us: 3 (0.05 ms), 30-100us: 1 (0.06 ms), 100-300us: 1 (0.13 ms), 300-1000000000us: 2 (61.69 ms)

== Agent 3: 129 tokens (delimiter void mul_mat_vec_q<(ggml_type)8, 1, false, false, false>(voi... grid 3973120), steady window 119 tokens, median per token:
  token period (delimiter to delimiter) 20.94 ms; inter-token gap 0.48 ms
  kernels 1466   span 20.46 ms   busy(union) 17.91 ms   idle-in-span 2.53 ms
  mmvq: 10.36 ms/433  mmq: 0.00 ms/0  fa: 0.40 ms/32  rccl: 2.65 ms/128  norm: 1.70 ms/209  cpy: 0.29 ms/70  other: 2.53 ms/594  rt: 0.00 ms/0
  gaps per token: 0-10us: 5 (0.03 ms), 10-30us: 3 (0.05 ms), 30-100us: 1 (0.09 ms), 100-300us: 0 (0.05 ms), 300-1000000000us: 2 (62.57 ms)

== Agent 4: 129 tokens (delimiter void mul_mat_vec_q<(ggml_type)8, 1, false, false, false>(voi... grid 3973120), steady window 119 tokens, median per token:
  token period (delimiter to delimiter) 20.94 ms; inter-token gap 0.52 ms
  kernels 1466   span 20.42 ms   busy(union) 17.08 ms   idle-in-span 3.35 ms
  mmvq: 10.53 ms/433  mmq: 0.00 ms/0  fa: 0.39 ms/32  rccl: 1.61 ms/128  norm: 1.74 ms/209  cpy: 0.28 ms/70  other: 2.52 ms/594  rt: 0.00 ms/0
  gaps per token: 0-10us: 5 (0.03 ms), 10-30us: 3 (0.05 ms), 30-100us: 1 (0.08 ms), 100-300us: 0 (0.04 ms), 300-1000000000us: 2 (79.92 ms)

"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"void ggml_cuda_tp::k_broadcast_reduce<4>(ggml_cuda_tp::RankData*, ggml_cuda_tp::RankSignals, ggml_cuda_tp::Signal*, float const*, float*, int, long)"
"void mul_mat_vec_q<(ggml_type)8, 1, false, false, false>(void const*, void const*, int const*, ggml_cuda_mm_fusion_args_device, float*, unsigned int,
"void mul_mat_vec_q<(ggml_type)8, 1, true, false, false>(void const*, void const*, int const*, ggml_cuda_mm_fusion_args_device, float*, unsigned int, 
"void rms_norm_f32<1024, true, false, true, true>(float const*, float*, int, long, long, long, float, float const*, long, long, long, HIP_vector_type<
"void quantize_q8_1<32>(float const*, void*, long, long, long, long, long, unsigned int, HIP_vector_type<unsigned int, 3u>)",66564,261118906,3922.8247
"void gated_delta_net_cuda<128, false, false>(float const*, float const*, float const*, float const*, float const*, float const*, float*, float*, long
"void flash_attn_tile<256, 256, 1, 2, false>(char const*, char const*, char const*, char const*, char const*, int const*, float*, HIP_vector_type<floa
