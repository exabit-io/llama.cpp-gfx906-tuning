patch:            substrate-def64b335 | mx-llama.cpp def64b335 | entangled (feature-group only) | mtp: validate deferred prefill replay
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
notes:            Touches: common/speculative.cpp, src/llama-context.cpp
                  Requires MTP / speculative decoding, which R3.9 puts out of this campaign.
                  Binned by INSTRUMENT TRIAGE, not by measurement, and the distinction matters: a flat
                  delta-minus reading against llama-batched-bench on Qwen3.8-27B would be evidence that
                  the instrument is blind, not that the commit is neutral. Five of the first six units
                  screened that way read flat for exactly this reason.
                  `neutral-required-substrate` because it stays in the substrate: dropping code that has
                  never been exercised is not justified by never having seen it do anything.
