patch:            substrate-cf6a98f73 | https://github.com/mxxm-t/mx-llama.cpp | cf6a98f73 | cuda: convert a large weight for cuBLAS in row blocks (bounded F32 workspace)
axis:             multi-user
zero point:       master 82868aa3b, read 2026-09-27 (round 2 per-group check, 0 GPU)
recipe:           this instrument: Qwen3.8-27B Q8_0, -sm tensor across four dies (tps 4, one stage), q8_repack on, no drafter
metric:           none — classified by reading the commit and its own measurements, not measured here
result:           not measured: inactive here: only the cuBLAS path with a converted 2D weight (e.g. a BF16 output head) changes; "the quantized kernels and every single-row decode are untouched" — the 27B Q8_0 runs q8_repack/MMQ
effect:           none measurable on this instrument
stats:            n/a (not measured) | n=0
evidence:         not-measurable
structural:       standalone
verdict:          untested
bin:              neutral-required-substrate
would change if:  a model with a large non-quantized head runs perplexity or multi-row logits through cuBLAS (it turns an OOM into a run)
notes:            Later mxxm-t commit listed in r2-preregistration.md as "classified by inspection". Applies equally to the
                  single-user axis (same one-stage tensor split). Stays in master with the substrate.
