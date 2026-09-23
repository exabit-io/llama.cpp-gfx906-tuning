patch:            substrate-481f684d3 | mx-llama.cpp 481f684d3 | delta-minus unit (separable) | cuda: admit only the GLU ops the repack fused epilogue implements
axis:             multi-user
zero point:       build-substrate-allquants 15.504 tok/s/slot, measured 2026-09-23 (same FA_QUANTS=all config, arms verified comparable)
recipe:           4x64K | -ctk q8_0 -ctv q8_0 | 125 W/die | --cache-ram 49152 | -ngl all | RCCL + custom AR ungated (the substrate lacks the gate knob, identically in both arms)
metric:           decode tok/s/slot with the commit REVERTED; a contribution shows as a DROP of >= 2%
result:           -0.18% against the substrate (n=2 per arm)
effect:           -0.18%
stats:            n=2 SCREEN; the permutation floor at n=2 vs n=2 is 0.333 so no q is attainable and none is claimed
evidence:         screened-only
structural:       standalone
verdict:          neutral
bin:              neutral-required-substrate
would change if:  a larger n or a different cell showed a contribution
notes:            Touches ggml/src/ggml-cuda (GLU/repack epilogue).
                  a general CUDA path the 27B DOES exercise. This is the one unit of the six where the flat reading is real evidence of no contribution at this cell.
                  METHODOLOGICAL POINT, and it applies to the whole substrate screen: a flat delta-minus
                  reading is only evidence of neutrality if the instrument EXECUTES the reverted code.
                  Five of the first six screened units fail that test on llama-batched-bench + the 27B.
                  Recording them as `neutral` would have been a false verdict of the same class that lost
                  the >300 tok/s configuration -- a result reported without naming the axis it was blind to.
                  Binned `neutral-required-substrate`: no evidence it contributes on this instrument, and
                  it stays in the substrate, because dropping code we cannot measure is not justified.
