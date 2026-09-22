patch:            q8-repack | mx-llama.cpp q8_repack/ (26 entangled commits + 2 separable) | Q8_0 weight repacking with --no-repack / -nr to disable | repacked narrow-batch mat-vec paths
axis:             multi-user
zero point:       build-substrate-v041 (ba82ea19a) measured 2026-09-20, --no-repack arm
recipe:           4 slots x 64K | q8_0 K and V | 125 W/die | --cache-ram 49152 | -ngl all | GGML_ENABLE_CUSTOM_AR=1, gate absent from this build
metric:           decode tok/s/slot at the 4x64K design point; improves requires q<0.10 and effect >= +2%
result:           multi 4x64K mean: ON 15.432 / OFF 14.636 tok/s/slot (n=4/n=5) | single 1x32K n=4 median: ON 39.686 / OFF 38.499 (full precision)
effect:           decode +5.44% multi-user, +3.08% single-user; prefill +5.24% multi, +2.79% single
stats:            multi decode p=0.00794 q=0.03175 | single decode p=0.05714 q=0.05714 | BH over the pre-registered family of 4 | n=4-5 fresh per arm per axis
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
