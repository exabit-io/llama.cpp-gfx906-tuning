patch:            substrate-51b31fc47 | mx-llama.cpp 51b31fc47 | entangled (feature-group only) | qwen4exp: shard the PLE gather table across the tensor-parallel group
axis:             multi-user
zero point:       none — not measured
recipe:           n/a; the available instrument cannot exercise this code
metric:           would be decode tok/s/slot, but only once an instrument that RUNS this path exists
result:           NOT MEASURED
effect:           unknown
stats:            n/a
evidence:         not-measurable
structural:       standalone
verdict:          untested
bin:              neutral-required-substrate
would change if:  measured with a Flash-Next measurement; the model is production per the lead 2026-09-23
notes:            Touches: ggml/src/ggml-backend-meta.cpp, ggml/src/ggml-cuda/getrows.cu, ggml/src/ggml-cuda/getrows.cuh ...
                  Requires Qwen3.8-Flash-Next (qwen4exp) — the 27B never takes this path.
                  Binned by INSTRUMENT TRIAGE, not by measurement, and the distinction matters: a flat
                  delta-minus reading against llama-batched-bench on Qwen3.8-27B would be evidence that
                  the instrument is blind, not that the commit is neutral. Five of the first six units
                  screened that way read flat for exactly this reason.
                  `neutral-required-substrate` because it stays in the substrate: dropping code that has
                  never been exercised is not justified by never having seen it do anything.
