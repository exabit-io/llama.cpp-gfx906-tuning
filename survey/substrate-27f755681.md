patch:            substrate-27f755681 | https://github.com/mxxm-t/mx-llama.cpp | 27f755681 | cuda: create the BLAS handle at backend init and report a full device at load instead of failing on the first prompt
axis:             multi-user
zero point:       master 82868aa3b, read 2026-09-27 (round 2 per-group check, 0 GPU)
recipe:           this instrument: Qwen3.8-27B Q8_0, -sm tensor across four dies (tps 4, one stage), q8_repack on, no drafter
metric:           none — classified by reading the commit and its own measurements, not measured here
result:           not measured: active here but a robustness change: its own guardrail on this instrument's model is "Qwen3.8-27B-Q8 4-GPU tensor -tps 4: 50.1 -> 50.2 t/s decode, identical output"; GGML_CUDA_EAGER_BLAS_HANDLE=0 restores the lazy handle
effect:           none measurable on this instrument
stats:            n/a (not measured) | n=0
evidence:         inspection
structural:       standalone
verdict:          untested
bin:              neutral-required-substrate
would change if:  a measurement shows the eager handle costing VRAM that a 64K+ layout needs (it moves the BLAS state in front of the reserve)
notes:            Later mxxm-t commit listed in r2-preregistration.md as "classified by inspection". Applies equally to the
                  single-user axis (same one-stage tensor split). Stays in master with the substrate.
