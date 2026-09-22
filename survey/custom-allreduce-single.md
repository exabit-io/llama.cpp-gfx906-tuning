patch:            custom-allreduce | mx-llama.cpp tp-allreduce.cu (1020 lines, 4 entangled commits) | GGML_ENABLE_CUSTOM_AR=1 | fork's own AllReduce for tensor-parallel reductions, PAIRED with tp-ar-size-gate
axis:             single-user
zero point:       build-c4series (bundle: substrate + 20 terms) measured 2026-09-21, GGML_ENABLE_CUSTOM_AR=0 arm — BOTH arms on ONE build so exactly one variable moves
recipe:           1 x 254K PRIMARY, 1 x 64K control | -ctk q8_0 | 125 W/die | --cache-ram 49152 | -ngl all | RCCL + gated custom AR
metric:           single-stream decode tok/s (prefill secondary), MEDIAN of n>=4 — the four-die single-stream stall fires about 1 run in 16 and shifts a mean by ~1.5%, most of the 2% floor; improves requires q<0.10 and effect >= +2%
result:           decode median 17.690 -> 18.578 (254K) and 31.687 -> 34.614 (64K) tok/s, n=6 per arm
effect:           decode +5.02% at 254K, +9.24% at 64K; prefill +/-0.06% (n.s.)
stats:            decode p=0.01299 q=0.02597 | BH over the pre-registered family | n=6 fresh per arm
evidence:         confirmed-fresh
structural:       required-by:tp-ar-size-gate
verdict:          improves
bin:              both
would change if:  a build-level removal of tp-allreduce.cu disagrees with the GGML_ENABLE_CUSTOM_AR=0 path, i.e. the env switch does not disable everything the 1020 lines add
notes:            Do not read the -19% prefill as a verdict against custom AllReduce. It is a verdict
                  against running it UNGATED: with tp-ar-size-gate applied, the same build keeps the
                  decode gain and loses none of the prefill (632.9 vs 633.1 with AR off entirely).
                  These two must therefore be binned as a PAIR, never independently -- which is what
                  'structural: required-by' records. Binning them separately would produce the exact
                  category error of rejecting a patch that is only bad in isolation.

confirmed 2026-09-21:   Custom AllReduce DOES earn its place, but only as a PAIR with the size gate.
                  Gated, it is +10.06% multi-user decode and +25.75% single-stream decode at zero
                  prefill cost. Ungated it costs -19% prefill (screen), which is why the pair must be
                  binned together: binning custom AR alone would have rejected it for a cost that only
                  exists in isolation. The +25.75% single-stream figure is also nearly double the
                  "+14% single-stream" the guide has carried since 2026-09-08.

                  METHOD NOTE, recorded because the mistake was nearly consequential. Reps 1-4 gave
                  these same effects on non-overlapping distributions and BOTH FAILED the pre-registered
                  test (p=0.0286 multi, 0.1429 single). I first diagnosed the median estimator as
                  under-powered at n=4 and was going to swap in a rank test plus a primary/secondary
                  endpoint hierarchy. A simulation refuted that diagnosis: median permutation reaches
                  p<0.10 in 100% of clean-separation trials at n=4. The real cause was TIES -- the
                  tool's rate columns print two decimals, one arm read 40.19 three times, and with a
                  repeated value a 3-1 split has the same median as the perfect 4-0 split.
                  Deriving the metric from the timing columns (three decimals) removes the ties and
                  every verdict then passes under the ORIGINAL rule. No post-hoc test change was kept.
                  Fix the instrument, not the test.

RE-BASED 2026-09-22 against RCCL:  The earlier +10.06%/+25.75% figures were against the BUTTERFLY
                  fallback, which R3.11 makes the wrong baseline. Against RCCL, custom AR still wins and
                  the verdict is unchanged in direction: +6.93% / +5.02% / +9.24% decode, with prefill
                  untouched. It is a decode optimisation and nothing else. The 1x32K single-user cells
                  that produced +25.75% are superseded: 32K is a floor, not a design point (R2.7).

AXIS NOTE:        companion record to custom-allreduce-ungated.md. R2.7 (corrected 2026-09-21) requires the single-user
                  axis to be measured at its OWN design point: 1 x 254K primary, 1 x 64K control.
                  Any single-user figure in this campaign dated before that correction was taken at
                  1 x 32K and is superseded -- a floor is not a design point.
