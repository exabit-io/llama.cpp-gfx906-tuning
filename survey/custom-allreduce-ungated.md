patch:            custom-allreduce | mx-llama.cpp tp-allreduce.cu (1020 lines, 4 entangled commits) | GGML_ENABLE_CUSTOM_AR=1 | fork's own AllReduce for tensor-parallel reductions
axis:             multi-user
zero point:       build-substrate-v041 (ba82ea19a) measured 2026-09-20, GGML_ENABLE_CUSTOM_AR=0 arm
recipe:           4 slots x 64K | q8_0 K and V | 125 W/die | --cache-ram 49152 | -ngl all | size gate NOT compiled into this build
metric:           decode tok/s/slot and prefill t/s at the 4x64K design point; improves requires q<0.10 and effect >= +2%
result:           screen n=2: AR ON 15.454 decode / 512.6 prefill; AR OFF 14.201 decode / 633.1 prefill
effect:           +8.82% decode, -19.03% prefill (screen), UNGATED
stats:            pending fresh confirmation; screen n=2 cannot produce a q value
evidence:         screened-only
structural:       required-by:tp-ar-size-gate
verdict:          unresolved
bin:              unbinned-pending
would change if:  the single-user arm shows the +14% single-stream gain the guide claims, which would make it valuable on that axis independently of the gate
notes:            Do not read the -19% prefill as a verdict against custom AllReduce. It is a verdict
                  against running it UNGATED: with tp-ar-size-gate applied, the same build keeps the
                  decode gain and loses none of the prefill (632.9 vs 633.1 with AR off entirely).
                  These two must therefore be binned as a PAIR, never independently -- which is what
                  'structural: required-by' records. Binning them separately would produce the exact
                  category error of rejecting a patch that is only bad in isolation.
