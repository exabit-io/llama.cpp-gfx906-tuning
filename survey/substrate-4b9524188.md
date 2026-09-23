patch:            substrate-4b9524188 | mx-llama.cpp 4b9524188 | delta-minus unit (separable) | meta: fold a reshaped split into dim 0 only for the gather output
axis:             multi-user
zero point:       build-substrate-allquants 15.504 tok/s/slot, measured 2026-09-23 (same FA_QUANTS=all config, arms verified comparable)
recipe:           4x64K | -ctk q8_0 -ctv q8_0 | 125 W/die | --cache-ram 49152 | -ngl all | RCCL + custom AR ungated (the substrate lacks the gate knob, identically in both arms)
metric:           decode tok/s/slot with the commit REVERTED; a contribution shows as a DROP of >= 2%
result:           +0.00% against the substrate (n=2 per arm)
effect:           +0.00%
stats:            n=2 SCREEN; the permutation floor at n=2 vs n=2 is 0.333 so no q is attainable and none is claimed
evidence:         not-measurable
structural:       standalone
verdict:          untested
bin:              neutral-required-substrate
would change if:  it were measured with an instrument that EXERCISES this path -- the server harness for server code, a sampling workload for top-k, or Qwen3.8-Flash-Next for qwen4exp and MoE routing
notes:            Touches ggml/src/ggml-backend-meta.cpp.
                  qwen4exp reshaped-split fold -- Flash-Next only, same as 31e3d2c5e.
                  METHODOLOGICAL POINT, and it applies to the whole substrate screen: a flat delta-minus
                  reading is only evidence of neutrality if the instrument EXECUTES the reverted code.
                  Five of the first six screened units fail that test on llama-batched-bench + the 27B.
                  Recording them as `neutral` would have been a false verdict of the same class that lost
                  the >300 tok/s configuration -- a result reported without naming the axis it was blind to.
                  Binned `neutral-required-substrate`: no evidence it contributes on this instrument, and
                  it stays in the substrate, because dropping code we cannot measure is not justified.
