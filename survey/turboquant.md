patch:            turboquant | arte-fact/llamacpp-gfx-906-turbo, stevio2d tq3_0, ggml discussion 21526 (HIP/ROCm port) | turbo3 KV compression: 3.5 bits/value, Walsh-Hadamard rotation + Lloyd-Max codebook
axis:             multi-user
zero point:       n/a — not ported
recipe:           n/a
metric:           would be decode tok/s/slot at 1x254K, where the KV term dominates the step
result:           not measured. Authors claim 4.6x vs f16 and ~18% TG cost; our own q4_0-V result suggests claimed TG costs on this class of change can be build-dependent
effect:           unknown
stats:            n/a
evidence:         inspection
structural:       standalone
verdict:          not-prioritised
bin:              technique-requires-implementation
would change if:  a round with budget for a ~1,800-line port, OR becoming memory-bound (a larger model such as Flash-Next at 103 GiB)
notes:            The 2026-09-09 survey deferred this with an explicit revisit condition -- "the 27B's KV is
                  small; revisit above ~100K per slot" -- and that condition is now MET: the design points
                  are 4x64K and 1x254K. The case for revisiting is no longer capacity (q8_0 KV at 254K is
                  only 2.1 GiB/die of a 31 GiB budget) but BANDWIDTH: the KV term is 94.6 ms/Mtok, over
                  half the step at 254K, and q4_0-V has now shown that cutting KV bytes converts directly
                  into decode speed on this build. Measure at 1x254K, never at 32K where the mechanism
                  cannot show.
