patch:            meta-xfer-rccl | https://github.com/mxxm-t/mx-llama.cpp | 04f89ab03 (part) | route stage-to-stage transfers through RCCL send/recv (GGML_META_XFER_RCCL)
axis:             multi-user
zero point:       master 82868aa3b, read 2026-09-25 (round 2 preparation, 0 GPU)
recipe:           this instrument: -sm tensor across all four dies = tps 4, n_stages 1 (ggml-backend-meta.cpp: n_stages = n_devs / tps)
metric:           none — the code path cannot execute on this instrument
result:           not measured: want_xfer_comm = n_stages > 1 && ... (ggml-backend-meta.cpp ~2951), so with one stage the XFER comm context is never created
effect:           none measurable here (the feature is inactive at n_stages = 1)
stats:            n/a (not measured) | n=0
evidence:         not-measurable
structural:       standalone
verdict:          untested
bin:              neutral-required-substrate
would change if:  the service runs more than one pipeline stage (-sm layer across stages, or tps < device count), where stage transfers exist
notes:            Round 2 pre-registration (r2-preregistration.md): dropped from the arms because its path needs n_stages > 1.
                  Applies equally to the single-user axis (same one-stage split). Stays in master with the substrate.
