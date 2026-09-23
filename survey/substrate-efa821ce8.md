patch:            substrate-efa821ce8 | mx-llama.cpp efa821ce8 | entangled (feature-group only) | cuda: hierarchical TOP_K past the shared-memory bitonic limit on HIP
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
would change if:  measured with bench/service-client.py or bench/chat-client.py against llama-server
notes:            Touches: common/debug.cpp, ggml/src/ggml-alloc.c, ggml/src/ggml-backend-meta.cpp ...
                  Requires the llama-server harness — llama-batched-bench is not the server.
                  Binned by INSTRUMENT TRIAGE, not by measurement, and the distinction matters: a flat
                  delta-minus reading against llama-batched-bench on Qwen3.8-27B would be evidence that
                  the instrument is blind, not that the commit is neutral. Five of the first six units
                  screened that way read flat for exactly this reason.
                  `neutral-required-substrate` because it stays in the substrate: dropping code that has
                  never been exercised is not justified by never having seen it do anything.
