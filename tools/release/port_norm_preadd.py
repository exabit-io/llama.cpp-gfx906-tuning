#!/usr/bin/env python3
# Port term 03 (residual ADD computed inside the fused RMS_NORM) onto the clamped-grid rms_norm_f32 kernel.
import sys, re
src, out = sys.argv[1], sys.argv[2]
s = open(src).read()

def rep(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        sys.exit(f"expected {count} of {old[:90]!r}, found {n}")
    s = s.replace(old, new)

rep("// gfx906 fusion 2026-09-08 (M1 trace: 257 quantize launches per decode token).\n"
    "template <int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false, bool clamped = false, bool do_q8 = false>",
    "// gfx906 fusion 2026-09-08 (M1 trace: 257 quantize launches per decode token).\n"
    "// do_preadd: the norm input is x + preadd_in (the residual add that precedes every pre-norm); the sum is written to\n"
    "// preadd_out (the residual stream, still needed downstream) and normalised from L2 in the same launch.\n"
    "template <int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false, bool clamped = false, bool do_q8 = false, bool do_preadd = false>")
rep("                                    const int64_t q8_stride_row        = 0) {\n    ggml_cuda_pdl_lc();",
    "                                    const int64_t q8_stride_row        = 0,\n"
    "                                    const float * preadd_in            = nullptr,\n"
    "                                    float *       preadd_out           = nullptr) {\n    ggml_cuda_pdl_lc();")
rep("""                q8c = q8_dst + ((int64_t) (sample*nchannels + channel)*nrows + row) * q8_stride_row;
            }

            float tmp = 0.0f; // partial sum for thread in warp

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                tmp += xi * xi;
            }
""", """                q8c = q8_dst + ((int64_t) (sample*nchannels + channel)*nrows + row) * q8_stride_row;
            }

            float tmp = 0.0f; // partial sum for thread in warp

            if constexpr (do_preadd) {
                // x, preadd_in and preadd_out share the (contiguous) layout of the add; each thread writes only the columns it
                // read, so an in-place add (preadd_out aliasing x or preadd_in) is safe.
                const float * preadd_inc  = preadd_in  + sample*stride_sample + channel*stride_channel + row*stride_row;
                float       * preadd_outc = preadd_out + sample*stride_sample + channel*stride_channel + row*stride_row;
                for (int col = tid; col < ncols; col += block_size) {
                    const float xi = xc[col] + preadd_inc[col];
                    preadd_outc[col] = xi;
                    tmp += xi * xi;
                }
                xc = preadd_outc; // the normalisation pass below re-reads the sum (L2-resident)
            } else {
                for (int col = tid; col < ncols; col += block_size) {
                    const float xi = xc[col];
                    tmp += xi * xi;
                }
            }
""")
# plain launches pass the new trailing arguments explicitly
rep("scale_out, (block_q8_1 *) nullptr, (int64_t) 0);", "scale_out, (block_q8_1 *) nullptr, (int64_t) 0, (const float *) nullptr, (float *) nullptr);", 2)
rep("1.0f, (block_q8_1 *) nullptr, (int64_t) 0);", "1.0f, (block_q8_1 *) nullptr, (int64_t) 0, (const float *) nullptr, (float *) nullptr);", 4)

# q8-only launch helper -> extended helper + run-time flag dispatch
i0 = s.index("template <int block_size, bool do_add, bool clamped>\nstatic void rms_norm_mul_q8_launch(")
i1 = s.index("static void rms_norm_mul_f32_cuda(const float *  x,")
s = s[:i0] + """template <int block_size, bool do_add, bool do_q8, bool do_preadd, bool clamped>
static void rms_norm_mul_ext_launch(const float * x, const float * mul, const float * add, float * dst, const int ncols,
                                    const int nchannels, const int nsamples,
                                    const dim3 & blocks_num, const int64_t stride_row, const int64_t stride_channel,
                                    const int64_t stride_sample, const float eps,
                                    const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
                                    const uint3 mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
                                    const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
                                    const uint3 add_ncols_packed, const uint3 add_nrows_packed, const uint3 add_nchannels_packed, const uint3 add_nsamples_packed,
                                    block_q8_1 * q8_dst, const int64_t q8_stride_row,
                                    const float * preadd_in, float * preadd_out, cudaStream_t stream) {
    const dim3 block_dims(block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
    ggml_cuda_kernel_launch(rms_norm_f32<block_size, true, do_add, false, clamped, do_q8, do_preadd>, launch_params,
        x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
        mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
        add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
        add_nchannels_packed, add_nsamples_packed, 1.0f, q8_dst, q8_stride_row, preadd_in, preadd_out);
}

// Dispatch over the run-time flags (bias add, q8 output, pre-add) for one block size.
template <int block_size, bool clamped>
static void rms_norm_mul_ext_dispatch(const bool has_add, const bool has_q8, const bool has_preadd,
                                      const float * x, const float * mul, const float * add, float * dst, const int ncols,
                                      const int nchannels, const int nsamples,
                                      const dim3 & blocks_num, const int64_t stride_row, const int64_t stride_channel,
                                      const int64_t stride_sample, const float eps,
                                      const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
                                      const uint3 mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
                                      const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
                                      const uint3 add_ncols_packed, const uint3 add_nrows_packed, const uint3 add_nchannels_packed, const uint3 add_nsamples_packed,
                                      block_q8_1 * q8_dst, const int64_t q8_stride_row, const float * preadd_in, float * preadd_out, cudaStream_t stream) {
#define RMS_NORM_EXT_CASE(A, Q, P) \\
    if (has_add == A && has_q8 == Q && has_preadd == P) { \\
        rms_norm_mul_ext_launch<block_size, A, Q, P, clamped>(x, mul, add, dst, ncols, nchannels, nsamples, blocks_num, stride_row, stride_channel, stride_sample, eps, \\
            mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, \\
            add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, \\
            q8_dst, q8_stride_row, preadd_in, preadd_out, stream); \\
        return; \\
    }
    RMS_NORM_EXT_CASE(false, true,  false)
    RMS_NORM_EXT_CASE(false, false, true)
    RMS_NORM_EXT_CASE(false, true,  true)
    RMS_NORM_EXT_CASE(true,  true,  false)
    RMS_NORM_EXT_CASE(true,  false, true)
    RMS_NORM_EXT_CASE(true,  true,  true)
#undef RMS_NORM_EXT_CASE
    GGML_ABORT("rms_norm_mul_ext_dispatch: no extended variant requested");
}

""" + s[i1:]

rep("""                                  block_q8_1 *   q8_dst        = nullptr,
                                  const int64_t  q8_stride_row = 0) {""",
"""                                  block_q8_1 *   q8_dst        = nullptr,
                                  const int64_t  q8_stride_row = 0,
                                  const float *  preadd_in     = nullptr,
                                  float *        preadd_out    = nullptr) {""")
rep("""        GGML_ASSERT(q8_dst == nullptr);
        rms_norm_f32_cuda(""", """        GGML_ASSERT(q8_dst == nullptr && preadd_in == nullptr);
        rms_norm_f32_cuda(""")
i0 = s.index("    if (q8_dst != nullptr) {\n        GGML_ASSERT(ncols % QK8_1 == 0);")
i1 = s.index("        return;\n    }\n    if (add == nullptr) {", i0)
s = s[:i0] + """    if (q8_dst != nullptr || preadd_in != nullptr) {
        GGML_ASSERT(q8_dst == nullptr || ncols % QK8_1 == 0);
        GGML_ASSERT((preadd_in == nullptr) == (preadd_out == nullptr));
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        const uint3 add_ncols_packed     = init_fastdiv_values(add ? add_ncols     : 1);
        const uint3 add_nrows_packed     = init_fastdiv_values(add ? add_nrows     : 1);
        const uint3 add_nchannels_packed = init_fastdiv_values(add ? add_nchannels : 1);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add ? add_nsamples  : 1);
        if (ncols < 1024) {
            rms_norm_mul_ext_dispatch<256, clamped>(add != nullptr, q8_dst != nullptr, preadd_in != nullptr, x, mul, add, dst, ncols, nchannels, nsamples, blocks_num, stride_row, stride_channel, stride_sample, eps,
                mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed,
                q8_dst, q8_stride_row, preadd_in, preadd_out, stream);
        } else {
            rms_norm_mul_ext_dispatch<1024, clamped>(add != nullptr, q8_dst != nullptr, preadd_in != nullptr, x, mul, add, dst, ncols, nchannels, nsamples, blocks_num, stride_row, stride_channel, stride_sample, eps,
                mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed,
                q8_dst, q8_stride_row, preadd_in, preadd_out, stream);
        }
""" + s[i1:]

rep("void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, void * q8_out) {",
    "// preadd: when non-null, the norm input dst->src[0] is the output of this ADD node, which the fused kernel computes\n"
    "// (src[0] + src[1], both contiguous and same-shaped) and writes on the way in.\n"
    "void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, void * q8_out, const ggml_tensor * preadd) {")
rep("""                                     ggml_tensor *               add_tensor,
                                     void *                      q8_out) {""",
"""                                     ggml_tensor *               add_tensor,
                                     void *                      q8_out,
                                     const ggml_tensor *         preadd) {""")
rep("                          eps, stream, (block_q8_1 *) q8_out, ne00 / QK8_1);\n}",
    "                          eps, stream, (block_q8_1 *) q8_out, ne00 / QK8_1,\n"
    "                          preadd ? (const float *) preadd->src[1]->data : nullptr, preadd ? (float *) preadd->data : nullptr);\n}", 2)
open(out, "w").write(s)
