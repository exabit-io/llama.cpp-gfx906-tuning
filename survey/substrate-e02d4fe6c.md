patch:            substrate-e02d4fe6c | mx-llama.cpp e02d4fe6c | separable | cuda: bound the top-k router matcher at the end of the graph
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
would change if:  measured with a sampling workload, or Flash-Next where MoE routing is on the decode path
notes:            Touches: ggml/src/ggml-cuda/ggml-cuda.cu
                  Requires MoE routing or a top-k sampler — batched-bench runs neither on the 27B.
                  Binned by INSTRUMENT TRIAGE, not by measurement, and the distinction matters: a flat
                  delta-minus reading against llama-batched-bench on Qwen3.8-27B would be evidence that
                  the instrument is blind, not that the commit is neutral. Five of the first six units
                  screened that way read flat for exactly this reason.
                  `neutral-required-substrate` because it stays in the substrate: dropping code that has
                  never been exercised is not justified by never having seen it do anything.
