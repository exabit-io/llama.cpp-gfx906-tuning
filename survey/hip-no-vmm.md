patch:            hip-no-vmm | build option GGML_HIP_NO_VMM=OFF | upstream | use the VMM memory pool instead of the legacy allocator
axis:             multi-user
zero point:       build-faq (NO_VMM=ON, the default) 2026-09-22
recipe:           n/a — never ran
metric:           would have been R3.2 memory headroom at 1x254K plus decode/prefill
result:           BUILD FAILS: ggml-cuda.cu:652 "unknown type name CUresult"; :653 "use of undeclared identifier CUresult"
effect:           not measurable
stats:            n/a
evidence:         not-measurable
structural:       standalone
verdict:          untested
bin:              technique-requires-implementation
would change if:  HIP gaining an equivalent of the CUDA VMM API (cuMemCreate/cuMemMap), or ggml adding a HIP path for the VMM pool
notes:            HIP has no CUDA virtual-memory-management API, so the VMM pool cannot compile on ROCm at all.
                  This is why the fork pins GGML_HIP_NO_VMM=ON. A failed build answered the unit at zero
                  GPU cost -- cheaper than the sweep I had budgeted for it.
