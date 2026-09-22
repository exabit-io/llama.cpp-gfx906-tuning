patch:            q8-repack | mx-llama.cpp q8_repack/ (26 entangled commits + 2 separable) | Q8_0 weight repacking with --no-repack / -nr to disable | repacked narrow-batch mat-vec paths
axis:             single-user
zero point:       build-substrate-v041 (ba82ea19a) measured 2026-09-20, --no-repack arm
recipe:           1 x 254K PRIMARY | -ctk q8_0 | 125 W/die | --cache-ram 49152 | -ngl all | RCCL + gated custom AR
metric:           single-stream decode tok/s (prefill secondary), MEDIAN of n>=4 — the four-die single-stream stall fires about 1 run in 16 and shifts a mean by ~1.5%, most of the 2% floor; improves requires q<0.10 and effect >= +2%
result:           decode median 16.395 -> 18.590 tok/s, prefill 368.3 -> 394.6 t/s (n=4)
effect:           decode +13.39%, prefill +7.13%
stats:            decode p=0.0571 q=0.0762, prefill q=0.0762 | BH over m=8 | n=4 fresh per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          improves
bin:              both
would change if:  a build-level revert of the 28 q8_repack commits disagrees with the --no-repack switch, which would mean the switch does not disable every path those commits added
notes:            Runtime-switchable via --no-repack, so it needs no build surgery -- which is why it
                  was screened in the first round. Positive on BOTH metrics at the design point, so it
                  is the strongest 'both' candidate so far, pending the single-user arm now running.
                  26 of its commits are entangled (later commits rewrote the same lines), so if the
                  runtime switch ever disagrees with a build-level revert, the switch is the weaker
                  instrument: it may not disable every path the commits added.

confirmed 2026-09-21:   Improves BOTH metrics on BOTH axes, every contrast through BH at q=0.10. It is
                  already inside the substrate and enabled by default, so the operational verdict is
                  "never pass --no-repack", not "apply a patch".
                  Caveat carried from the record above: the instrument is a runtime switch, and 26 of
                  its 28 commits are entangled. If a build-level revert ever disagrees with the switch,
                  the build-level result wins.

full-precision re-derivation 2026-09-21: verdict unchanged, q slightly stronger. The metric is now
                  derived from the timing columns rather than the tool's 2-decimal rate columns; see
                  custom-allreduce-ungated.md for why that mattered. One extra repack-off cell (n=5) is
                  a complete 489-byte table from the run that was killed on 2026-09-20 before its TSV
                  row was appended -- included on a completeness test, not on its value, and it agrees
                  with the independent re-measurement to 0.06%.

PROVISIONAL 2026-09-22:  Measured on a build WITHOUT RCCL (butterfly collective) and with its
                  single-user arm at 1x32K. Both are now wrong per R3.11 and R2.7. The repack effect is
                  a weight-layout change and is not expected to interact with the collective, but that is
                  an expectation, not a measurement. Re-measure on build-faq (RCCL + corrected FA_QUANTS)
                  at 4x64K and 1x254K before this verdict is treated as final.

CLOSED 2026-09-22:  Re-measured on the shipping collective (RCCL) at the corrected design points, which
                  is what the earlier PROVISIONAL note demanded. The effect is LARGER than the
                  butterfly-based numbers suggested and it lands on different metrics at different
                  shapes: prefill at 4x64K (+13.59%) and decode at 1x254K (+13.39%). A single design
                  point would have hidden that, which is the argument for the two-depth requirement in
                  miniature.
                  Bin BOTH. Already default-on inside the substrate, so the operational rule is simply:
                  never pass --no-repack. The 4x64K decode figure (+1.97%) sits just under the 2%
                  materiality floor and is reported as such rather than rounded up.

AXIS NOTE:        companion record to q8-repack.md. R2.7 (corrected 2026-09-21) requires the single-user
                  axis to be measured at its OWN design point: 1 x 254K primary, 1 x 64K control.
                  Any single-user figure in this campaign dated before that correction was taken at
                  1 x 32K and is superseded -- a floor is not a design point.
