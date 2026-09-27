patch:            substrate-be8ff98aa | https://github.com/mxxm-t/mx-llama.cpp | be8ff98aa | meta: run a node on the stage that holds the stage-local mirrored weights it reads
axis:             multi-user
zero point:       master 82868aa3b, read 2026-09-27 (round 2 per-group check, 0 GPU)
recipe:           this instrument: Qwen3.8-27B Q8_0, -sm tensor across four dies (tps 4, one stage), q8_repack on, no drafter
metric:           none — classified by reading the commit and its own measurements, not measured here
result:           not measured: inactive here: stage placement needs n_stages > 1; this instrument runs one stage of four (tps 4)
effect:           none measurable on this instrument
stats:            n/a (not measured) | n=0
evidence:         not-measurable
structural:       standalone
verdict:          untested
bin:              neutral-required-substrate
would change if:  a multi-stage split (tps < devices), e.g. a draft context with fewer stages than its target
notes:            Later mxxm-t commit listed in r2-preregistration.md as "classified by inspection". Applies equally to the
                  single-user axis (same one-stage tensor split). Stays in master with the substrate.
