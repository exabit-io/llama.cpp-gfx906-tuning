# qwen38-27b-fa-counters-2  2026-09-08T17:33:58+00:00  FA head-256 counters (gfx906 counter set), production build tp4

## prefill: kernel time share (pass A trace), then counters per dispatch, all dies summed
| kernel | time % | dispatches | us/dispatch | VALU busy % | waves/dispatch | VALU inst/dispatch | VMEM_RD | LDS inst | LDS wait / GUI_ACTIVE | LDS bank confl | L2 hit % | SALU/VALU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| void mul_mat_q<(ggml_type)8, 128, false>(char const*, int const* | 62.4 | 47104 | 3380 | 61 | 3.96e+03 | 2.24e+08 | 2.31e+06 | 4.29e+07 | 2.399 | 6.15e+06 | 79 | 0.01 |
| ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>) | 12.0 | 16896 | 1812 | 1 | 63.8 | 1.45e+06 | 7.36e+04 | 2.85e+04 | 0.000 | 0 | 8 | 0.26 |
| void flash_attn_tile<256, 256, 16, 2, false>(char const*, char c | 9.0 | 2112 | 10940 | 46 | 2.98e+03 | 5.34e+08 | 3.9e+06 | 9.19e+07 | 7.491 | 0 | 94 | 0.01 |
| void gated_delta_net_cuda<128, false, false>(float const*, float | 6.2 | 6528 | 2430 | 58 | 1.67e+03 | 1.39e+08 | 2.07e+07 | 3.56e+07 | 10.656 | 0 | 96 | 0.34 |
| void mul_mat_q<(ggml_type)8, 64, true>(char const*, int const*,  | 1.4 | 12288 | 284 | 29 | 256 | 9.14e+06 | 1.43e+05 | 1.93e+06 | 1.723 | 4.61e+04 | 25 | 0.03 |
| void k_bin_bcast<&(op_add(float, float)), float, float, float, f | 1.2 | 22912 | 130 | 61 | 5.86e+04 | 6.57e+06 | 2.35e+05 | 0 | 0.000 | 0 | 33 | 0.28 |
| void quantize_mmq_q8_1<(mmq_q8_1_ds_layout)0, false>(float const | 1.2 | 32768 | 90 | 44 | 3.28e+04 | 3.79e+06 | 3.23e+04 | 9.83e+04 | 0.031 | 0 | 41 | 0.45 |
| void concat_non_cont<unsigned int, 0>(char const*, char const*,  | 1.1 | 6336 | 463 | 5 | 1.02e+04 | 1.91e+06 | 8.2e+04 | 0 | 0.000 | 0 | 86 | 0.57 |
| void rms_norm_f32<1024, true, false, false, false>(float const*, | 1.1 | 16512 | 163 | 36 | 3.28e+04 | 5.73e+06 | 4.92e+05 | 3.93e+05 | 1.224 | 0 | 40 | 0.65 |
| void convert_unary<float, __hip_bfloat16>(void const*, __hip_bfl | 1.0 | 16384 | 148 | 46 | 1.64e+05 | 6.39e+06 | 1.64e+05 | 0 | 0.000 | 0 | 0 | 1.97 |
| void convert_unary<__hip_bfloat16, float>(void const*, float*, l | 0.9 | 16384 | 147 | 42 | 1.64e+05 | 5.73e+06 | 1.64e+05 | 0 | 0.000 | 0 | 0 | 2.17 |
| void unary_gated_op_kernel<&(op_silu(float)), float>(float const | 0.6 | 15232 | 104 | 67 | 9.48e+04 | 5.69e+06 | 1.9e+05 | 0 | 0.000 | 0 | 7 | 0.15 |

## decode: kernel time share (pass A trace), then counters per dispatch, all dies summed
| kernel | time % | dispatches | us/dispatch | VALU busy % | waves/dispatch | VALU inst/dispatch | VMEM_RD | LDS inst | LDS wait / GUI_ACTIVE | LDS bank confl | L2 hit % | SALU/VALU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| void mul_mat_q<(ggml_type)8, 128, false>(char const*, int const* | 57.9 | 94208 | 3381 | 61 | 3.96e+03 | 2.24e+08 | 2.31e+06 | 4.29e+07 | 2.401 | 6.15e+06 | 79 | 0.01 |
| void flash_attn_tile<256, 256, 16, 2, false>(char const*, char c | 16.6 | 4160 | 21963 | 47 | 3.03e+03 | 1.07e+09 | 7.83e+06 | 1.85e+08 | 7.648 | 0 | 95 | 0.01 |
| ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>) | 9.6 | 33280 | 1578 | 1 | 63.9 | 1.41e+06 | 7.53e+04 | 2.95e+04 | 0.000 | 0 | 8 | 0.27 |
| void gated_delta_net_cuda<128, false, false>(float const*, float | 5.8 | 15548 | 2043 | 58 | 2.45e+03 | 1.17e+08 | 1.74e+07 | 2.99e+07 | 10.714 | 0 | 96 | 0.34 |
| void mul_mat_q<(ggml_type)8, 64, true>(char const*, int const*,  | 1.3 | 24576 | 284 | 29 | 256 | 9.14e+06 | 1.43e+05 | 1.93e+06 | 1.724 | 4.61e+04 | 25 | 0.03 |
| void k_bin_bcast<&(op_add(float, float)), float, float, float, f | 1.1 | 48312 | 122 | 61 | 5.56e+04 | 6.23e+06 | 2.22e+05 | 0 | 0.000 | 0 | 33 | 0.28 |
| void quantize_mmq_q8_1<(mmq_q8_1_ds_layout)0, false>(float const | 1.1 | 65536 | 90 | 44 | 3.28e+04 | 3.79e+06 | 3.23e+04 | 9.83e+04 | 0.031 | 0 | 41 | 0.45 |
| void concat_non_cont<unsigned int, 0>(char const*, char const*,  | 1.1 | 12480 | 470 | 5 | 1.02e+04 | 1.94e+06 | 8.32e+04 | 0 | 0.000 | 0 | 86 | 0.57 |
| void rms_norm_f32<1024, true, false, false, false>(float const*, | 1.0 | 33024 | 162 | 36 | 3.28e+04 | 5.73e+06 | 4.92e+05 | 3.93e+05 | 1.222 | 0 | 40 | 0.65 |
| void convert_unary<float, __hip_bfloat16>(void const*, __hip_bfl | 0.9 | 32768 | 148 | 46 | 1.64e+05 | 6.39e+06 | 1.64e+05 | 0 | 0.000 | 0 | 0 | 1.97 |
| void convert_unary<__hip_bfloat16, float>(void const*, float*, l | 0.9 | 32768 | 147 | 42 | 1.64e+05 | 5.73e+06 | 1.64e+05 | 0 | 0.000 | 0 | 0 | 2.17 |
| void unary_gated_op_kernel<&(op_silu(float)), float>(float const | 0.6 | 36280 | 87 | 66 | 7.96e+04 | 4.78e+06 | 1.59e+05 | 0 | 0.000 | 0 | 7 | 0.15 |
# done 2026-09-08T18:00:37+00:00
