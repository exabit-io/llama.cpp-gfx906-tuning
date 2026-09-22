patch:            fa-quants-dead-combos | GGML_CUDA_FA_QUANTS defaults bf16-bf16 and q4_0-q4_0 | upstream defaults | attention kernels compiled for combinations this hardware cannot use
axis:             multi-user
zero point:       upstream default FA_QUANTS list
recipe:           build-time only
metric:           binary size and compile time; no runtime metric — the kernels are never selected here
result:           removed from the campaign value; build unaffected at runtime, one 448 KB object and its compile time saved
effect:           no runtime effect
stats:            n/a
evidence:         inspection
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  hardware with native bf16 (RDNA3+/CDNA/Ampere+) or a use case wanting a 4-bit KEY cache
notes:            bf16 is unusable twice over here: fast_bf16_hardware_available requires RDNA3+ or CDNA and
                  gfx906 is GCN5, while the W-3275M host is Cascade Lake-SP with no AVX512-BF16. Nothing to
                  do with MKL -- GGML_BLAS is OFF entirely. q4_0-q4_0 is a 4-bit KEY cache, which no
                  requirement asks for. Their slots in the list displaced q8_0-q4_0, the combination R2.2
                  names as the route to 256K; see q4v-cache.md for what that cost.
