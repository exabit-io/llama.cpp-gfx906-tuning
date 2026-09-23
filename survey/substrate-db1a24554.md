patch:            substrate-db1a24554 | mx-llama.cpp db1a24554 | entangled (feature-group only) | spec: defer DSpark prefill across server slots
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
would change if:  measured with the MTP gate, once R3.9 is in scope
notes:            Touches: common/speculative.cpp, ggml/src/ggml-backend-meta.cpp, src/llama-context.cpp ...
                  Requires MTP / speculative decoding, which R3.9 puts out of this campaign.
                  Binned by INSTRUMENT TRIAGE, not by measurement, and the distinction matters: a flat
                  delta-minus reading against llama-batched-bench on Qwen3.8-27B would be evidence that
                  the instrument is blind, not that the commit is neutral. Five of the first six units
                  screened that way read flat for exactly this reason.
                  `neutral-required-substrate` because it stays in the substrate: dropping code that has
                  never been exercised is not justified by never having seen it do anything.
