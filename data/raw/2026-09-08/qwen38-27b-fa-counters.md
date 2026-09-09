# qwen38-27b-fa-counters  2026-09-08T13:53:10+00:00  FA head-256 counters, production build tp4

## prefill: counter files 1
| kernel | SQ_BUSY_CYCLES | SQ_INSTS_VALU | VALU per busy cycle | SQ_WAIT_INST_ANY / busy | TCP accesses |
|---|---:|---:|---:|---:|---:|
| void mul_mat_vec_q<(ggml_type)8, 4, false, false>(void const*, void co | 0 | 1.62e+09 | 0.000 | 0.000 | 0 |
| void mul_mat_vec_q<(ggml_type)8, 16, false, false>(void const*, void c | 0 | 6.53e+09 | 0.000 | 0.000 | 0 |
| void mul_mat_vec_q<(ggml_type)8, 1, false, false>(void const*, void co | 0 | 5.59e+08 | 0.000 | 0.000 | 0 |
| void mul_mat_q<(ggml_type)8, 64, true>(char const*, int const*, int co | 0 | 2.25e+11 | 0.000 | 0.000 | 0 |
| void mul_mat_q<(ggml_type)8, 64, false>(char const*, int const*, int c | 0 | 1.5e+11 | 0.000 | 0.000 | 0 |
| void mul_mat_q<(ggml_type)8, 128, false>(char const*, int const*, int  | 0 | 2.11e+13 | 0.000 | 0.000 | 0 |
| void flash_attn_tile<256, 256, 16, 2, false>(char const*, char const*, | 0 | 4.47e+12 | 0.000 | 0.000 | 0 |
| void flash_attn_tile<256, 256, 1, 2, false>(char const*, char const*,  | 0 | 3.8e+08 | 0.000 | 0.000 | 0 |

## decode: counter files 1
| kernel | SQ_BUSY_CYCLES | SQ_INSTS_VALU | VALU per busy cycle | SQ_WAIT_INST_ANY / busy | TCP accesses |
|---|---:|---:|---:|---:|---:|
| void mul_mat_vec_q<(ggml_type)8, 1, true, false>(void const*, void con | 0 | 1.27e+10 | 0.000 | 0.000 | 0 |
| void mul_mat_vec_q<(ggml_type)8, 1, false, false>(void const*, void co | 0 | 2.23e+10 | 0.000 | 0.000 | 0 |
| void mul_mat_q<(ggml_type)8, 88, false>(char const*, int const*, int c | 0 | 1.37e+12 | 0.000 | 0.000 | 0 |
| void mul_mat_q<(ggml_type)8, 64, true>(char const*, int const*, int co | 0 | 2.25e+11 | 0.000 | 0.000 | 0 |
| void mul_mat_q<(ggml_type)8, 64, false>(char const*, int const*, int c | 0 | 1.5e+11 | 0.000 | 0.000 | 0 |
| void mul_mat_q<(ggml_type)8, 128, false>(char const*, int const*, int  | 0 | 1.98e+13 | 0.000 | 0.000 | 0 |
| void flash_attn_tile<256, 256, 16, 2, false>(char const*, char const*, | 0 | 1.78e+13 | 0.000 | 0.000 | 0 |
| void flash_attn_tile<256, 256, 1, 2, false>(char const*, char const*,  | 0 | 1.21e+10 | 0.000 | 0.000 | 0 |

## kernel-time share (from the kernel stats of each pass)
### prefill
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"void mul_mat_q<(ggml_type)8, 128, false>(char const*, int const*, int const*, int const*, float*, float*, float const*, HIP_vector_type<unsigned int, 3u>, int,
"void flash_attn_tile<256, 256, 16, 2, false>(char const*, char const*, char const*, char const*, char const*, int const*, float*, HIP_vector_type<float, 2u>*, 
"ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)",33792,55243789227,1634818.573242,10.06,58080,1199724305,11522962.587869
"void gated_delta_net_cuda<128, false, false>(float const*, float const*, float const*, float const*, float const*, float const*, float*, float*, long, long, lo
"void mul_mat_q<(ggml_type)8, 64, true>(char const*, int const*, int const*, int const*, float*, float*, float const*, HIP_vector_type<unsigned int, 3u>, int, i
"void quantize_mmq_q8_1<(mmq_q8_1_ds_layout)0, false>(float const*, int const*, void*, long, long, long, long, long, int, int, int)",65536,5891981361,89904.5007
"void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(float const*, float const*, float*, unsigned int, unsigned int, unsigned int, HIP_
"void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, long, long, long, unsigned long, unsigned long, unsigned long, unsigned long, long
"void rms_norm_f32<1024, true, false, false, false>(float const*, float*, int, long, long, long, float, float const*, long, long, long, HIP_vector_type<unsigned
"void convert_unary<float, __hip_bfloat16>(void const*, __hip_bfloat16*, long, long, long, HIP_vector_type<unsigned int, 3u>, long, long, long)",32768,485601537
"void convert_unary<__hip_bfloat16, float>(void const*, float*, long, long, long, HIP_vector_type<unsigned int, 3u>, long, long, long)",32768,4816771137,146996.
### decode
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"void mul_mat_q<(ggml_type)8, 128, false>(char const*, int const*, int const*, int const*, float*, float*, float const*, HIP_vector_type<unsigned int, 3u>, int,
"void flash_attn_tile<256, 256, 16, 2, false>(char const*, char const*, char const*, char const*, char const*, int const*, float*, HIP_vector_type<float, 2u>*, 
"ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)",147968,144315697802,975316.945569,13.55,29760,9890954,1355005.706164
"void mul_mat_q<(ggml_type)8, 88, false>(char const*, int const*, int const*, int const*, float*, float*, float const*, HIP_vector_type<unsigned int, 3u>, int, 
"void gated_delta_net_cuda<128, false, false>(float const*, float const*, float const*, float const*, float const*, float const*, float*, float*, long, long, lo
"void mul_mat_q<(ggml_type)8, 64, true>(char const*, int const*, int const*, int const*, float*, float*, float const*, HIP_vector_type<unsigned int, 3u>, int, i
"void mul_mat_q<(ggml_type)8, 64, false>(char const*, int const*, int const*, int const*, float*, float*, float const*, HIP_vector_type<unsigned int, 3u>, int, 
"void rms_norm_f32<1024, true, false, false, false>(float const*, float*, int, long, long, long, float, float const*, long, long, long, HIP_vector_type<unsigned
"void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, long, long, long, unsigned long, unsigned long, unsigned long, unsigned long, long
"void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(float const*, float const*, float*, unsigned int, unsigned int, unsigned int, HIP_
"void quantize_mmq_q8_1<(mmq_q8_1_ds_layout)0, false>(float const*, int const*, void*, long, long, long, long, long, int, int, int)",262144,5800928845,22128.787
# done 2026-09-08T14:09:12+00:00
