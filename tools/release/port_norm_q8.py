#!/usr/bin/env python3
# Port term 02 (fused RMS_NORM+MUL[+ADD] also emits Q8_1) onto the clamped-grid rms_norm_f32 kernel.
import sys
src, out = sys.argv[1], sys.argv[2]
s = open(src).read()

def rep(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        sys.exit(f"expected {count} of {old[:80]!r}, found {n}")
    s = s.replace(old, new)

rep("template <int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false, bool clamped = false>\nstatic __global__ void rms_norm_f32(",
    "// do_q8: also write the (normalised, scaled) row as Q8_1 blocks in the MMVQ row-wise layout (one block per 32 columns,\n"
    "// ncols == padded row length), so the quantised matrix-vector consumers of this norm skip their quantize launch.\n"
    "// gfx906 fusion 2026-09-08 (M1 trace: 257 quantize launches per decode token).\n"
    "template <int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false, bool clamped = false, bool do_q8 = false>\nstatic __global__ void rms_norm_f32(")
rep("                                    const float   scale_out            = 1.0f) {\n    ggml_cuda_pdl_lc();\n    const int nrows = gridDim.x;",
    "                                    const float   scale_out            = 1.0f,\n"
    "                                    block_q8_1 *  q8_dst               = nullptr,\n"
    "                                    const int64_t q8_stride_row        = 0) {\n    ggml_cuda_pdl_lc();\n    const int nrows = gridDim.x;")
rep('    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");\n\n    extern __shared__ float s_sum[];\n\n    ggml_cuda_pdl_sync();\n\n    // grid.y and grid.z are clamped',
    '    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");\n'
    '    static_assert(!do_q8 || do_multiply, "q8 output is only wired through the fused multiply path");\n'
    '    static_assert(!do_q8 || block_size % QK8_1 == 0, "q8 output needs whole 32-lane groups per block");\n\n'
    '    extern __shared__ float s_sum[];\n\n    ggml_cuda_pdl_sync();\n\n    // grid.y and grid.z are clamped')
rep("                addc = add + add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;\n            }\n\n            float tmp = 0.0f; // partial sum for thread in warp\n",
    "                addc = add + add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;\n            }\n\n"
    "            [[maybe_unused]] block_q8_1 * q8c = nullptr;\n"
    "            if constexpr (do_q8) {\n"
    "                // ncols % QK8_1 == 0 (checked by the caller): every 32-lane group is either fully inside or fully outside\n"
    "                // the row on each loop trip, so the width-32 shuffles below never see a diverged group.\n"
    "                q8c = q8_dst + ((int64_t) (sample*nchannels + channel)*nrows + row) * q8_stride_row;\n"
    "            }\n\n"
    "            float tmp = 0.0f; // partial sum for thread in warp\n")
rep("""            for (int col = tid; col < ncols; col += block_size) {
                if constexpr (do_multiply && do_add) {
                    const int mul_col = fastmodulo(col, mul_ncols_packed);
                    const int add_col = fastmodulo(col, add_ncols_packed);
                    dstc[col]         = scale * xc[col] * mulc[mul_col] + addc[add_col];
                } else if constexpr (do_multiply) {
                    const int mul_col = fastmodulo(col, mul_ncols_packed);
                    dstc[col]         = scale * xc[col] * mulc[mul_col];
                } else if constexpr (do_scale) {
                    dstc[col] = scale_out * (scale * xc[col]);
                } else {
                    dstc[col] = scale * xc[col];
                }
            }
""", """            for (int col = tid; col < ncols; col += block_size) {
                float v;
                if constexpr (do_multiply && do_add) {
                    const int mul_col = fastmodulo(col, mul_ncols_packed);
                    const int add_col = fastmodulo(col, add_ncols_packed);
                    v                 = scale * xc[col] * mulc[mul_col] + addc[add_col];
                } else if constexpr (do_multiply) {
                    const int mul_col = fastmodulo(col, mul_ncols_packed);
                    v                 = scale * xc[col] * mulc[mul_col];
                } else if constexpr (do_scale) {
                    v = scale_out * (scale * xc[col]);
                } else {
                    v = scale * xc[col];
                }
                dstc[col] = v;
                if constexpr (do_q8) {
                    // Same arithmetic as quantize_q8_1: d = amax/127, q = round(v/d), ds = (d, sum of the fp32 values).
                    const float  amax = warp_reduce_max<QK8_1>(fabsf(v));
                    const float  sum  = warp_reduce_sum<QK8_1>(v);
                    const float  d    = amax / 127.0f;
                    const int8_t q    = amax == 0.0f ? 0 : roundf(v / d);
                    block_q8_1 * y    = q8c + col / QK8_1;
                    y->qs[col % QK8_1] = q;
                    if (col % QK8_1 == 0) {
                        y->ds = make_half2(d, sum);
                    }
                }
            }
""")
# every launch passes the trailing kernel arguments explicitly (cudaLaunchKernelEx has no default params)
rep("make_uint3(0, 0, 0), scale_out);", "make_uint3(0, 0, 0), scale_out, (block_q8_1 *) nullptr, (int64_t) 0);", 2)
rep("make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f);", "make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f, (block_q8_1 *) nullptr, (int64_t) 0);", 2)
rep("add_nchannels_packed, add_nsamples_packed, 1.0f);", "add_nchannels_packed, add_nsamples_packed, 1.0f, (block_q8_1 *) nullptr, (int64_t) 0);", 2)

rep("static void rms_norm_mul_f32_cuda(const float *  x,",
"""template <int block_size, bool do_add, bool clamped>
static void rms_norm_mul_q8_launch(const float * x, const float * mul, const float * add, float * dst, const int ncols,
                                   const int nchannels, const int nsamples,
                                   const dim3 & blocks_num, const int64_t stride_row, const int64_t stride_channel,
                                   const int64_t stride_sample, const float eps,
                                   const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
                                   const uint3 mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
                                   const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
                                   const uint3 add_ncols_packed, const uint3 add_nrows_packed, const uint3 add_nchannels_packed, const uint3 add_nsamples_packed,
                                   block_q8_1 * q8_dst, const int64_t q8_stride_row, cudaStream_t stream) {
    const dim3 block_dims(block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
    ggml_cuda_kernel_launch(rms_norm_f32<block_size, true, do_add, false, clamped, true>, launch_params,
        x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
        mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
        add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
        add_nchannels_packed, add_nsamples_packed, 1.0f, q8_dst, q8_stride_row);
}

static void rms_norm_mul_f32_cuda(const float *  x,""")
rep("""                                  const float    eps,
                                  cudaStream_t   stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));""",
"""                                  const float    eps,
                                  cudaStream_t   stream,
                                  block_q8_1 *   q8_dst        = nullptr,
                                  const int64_t  q8_stride_row = 0) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));""")
rep("""    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {""",
"""    if (mul == nullptr) {
        GGML_ASSERT(q8_dst == nullptr);
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (q8_dst != nullptr) {
        GGML_ASSERT(ncols % QK8_1 == 0);
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        const uint3 add_ncols_packed     = init_fastdiv_values(add ? add_ncols     : 1);
        const uint3 add_nrows_packed     = init_fastdiv_values(add ? add_nrows     : 1);
        const uint3 add_nchannels_packed = init_fastdiv_values(add ? add_nchannels : 1);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add ? add_nsamples  : 1);
        if (add == nullptr) {
            if (ncols < 1024) {
                rms_norm_mul_q8_launch<256, false, clamped>(x, mul, nullptr, dst, ncols, nchannels, nsamples, blocks_num, stride_row, stride_channel, stride_sample, eps,
                    mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    0, 0, 0, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, q8_dst, q8_stride_row, stream);
            } else {
                rms_norm_mul_q8_launch<1024, false, clamped>(x, mul, nullptr, dst, ncols, nchannels, nsamples, blocks_num, stride_row, stride_channel, stride_sample, eps,
                    mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    0, 0, 0, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, q8_dst, q8_stride_row, stream);
            }
        } else {
            if (ncols < 1024) {
                rms_norm_mul_q8_launch<256, true, clamped>(x, mul, add, dst, ncols, nchannels, nsamples, blocks_num, stride_row, stride_channel, stride_sample, eps,
                    mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, q8_dst, q8_stride_row, stream);
            } else {
                rms_norm_mul_q8_launch<1024, true, clamped>(x, mul, add, dst, ncols, nchannels, nsamples, blocks_num, stride_row, stride_channel, stride_sample, eps,
                    mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, q8_dst, q8_stride_row, stream);
            }
        }
        return;
    }
    if (add == nullptr) {""")
rep("void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {",
    "void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, void * q8_out) {")
rep("""                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {""",
"""                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor,
                                     void *                      q8_out) {""")
rep("                          eps, stream);\n}", "                          eps, stream, (block_q8_1 *) q8_out, ne00 / QK8_1);\n}", 2)
open(out, "w").write(s)
