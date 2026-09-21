patch:            tp-ar-size-gate | exabit-io/llama.cpp terms/06-39e3ad8bb | GGML_TP_AR_MAX_NE explicit size gate | restrict the fork's custom AllReduce to tensors at or below N elements
axis:             multi-user
zero point:       build-c4series (substrate+20 terms, f04198b3f lineage) measured 2026-09-20, gate OFF arm
recipe:           4 slots x 64K | q8_0 K and V | 125 W/die | --cache-ram 49152 | -ngl all | env stated per arm, gfx906.env NOT sourced
metric:           prefill t/s at the 4x64K design point; improves requires q<0.10 and effect >= +2%
result:           multi 4x64K n=4: gate ON 633.27 +/- 0.47 / OFF 511.36 +/- 1.97 t/s prefill; single 1x32K n=4 median: ON 701.02 / OFF 546.12 t/s
effect:           prefill +23.84% multi-user, +28.36% single-user; decode -0.04% multi / +0.19% single (both n.s.)
stats:            multi prefill p=0.0286 q=0.0761 PASS | single prefill p=0.0571 q=0.0761 PASS | multi decode p=0.9429 q=0.9429 n.s. | single decode p=0.7429 q=0.8490 n.s. | BH across m=8 tests at q<0.10 | n=4 fresh per arm per axis
evidence:         confirmed-fresh
structural:       standalone
verdict:          improves
bin:              both
would change if:  the prefill gain turns out to depend on --cache-ram 49152 rather than the gate, or a different n_embd/TP width moves the 20481 threshold off the 4-row boundary
notes:            THIS RECORD EXISTS TO CORRECT AN ATTRIBUTION, and the correction is what makes it
                  important rather than the effect size. Since 2026-09-08 the guide has credited the
                  fork's tile table with "+33% prefill" -- by inspection only, as CLAUDE.md states.
                  Three independently measured configurations now reconcile differently:
                    stock v0.4.1                      522.9 t/s
                    substrate, custom AR ungated      510.2  (gate 1, 2026-09-20)
                    substrate, custom AR OFF          633.1  (E1 screen)
                    bundle, gate ON                   632.9  (E3 screen)
                    bundle, gate OFF                  510.8  (E3 screen)
                  The substrate's prefill machinery (tile table and friends) is worth ~+21% over stock;
                  UNGATED custom AllReduce was masking all of it at -19%. What delivers prefill in a
                  running configuration is therefore this one-line size gate, not the tile table. The
                  tile table's own value must be measured separately with AR held off.
                  Mechanism, consistent with the numbers: prefill works on large tensors, so the gate
                  routes them to the standard path; decode works on narrow ones (<= 4 rows) which stay
                  on custom AR and keep its +8.8%. The gate gets both.
                  Implication for the survey: a term's effect can depend on an env var, so env must be
                  stated per arm and never inherited. The substrate binary does not even compile in
                  GGML_TP_AR_MAX_NE, so every run before 2026-09-20 14:00 set an env var that only the
                  bundle build could honour.

confirmed 2026-09-21:   The fresh n=4 confirmation reproduces the screen almost exactly (+23.84% vs
                  +23.90% screened). State continuity was checked before pooling across two days: the
                  rep-4 gate-on cell read 632.68 t/s against 633.11 / 633.68 / 633.63 banked the day
                  before, 0.15% apart, so the dies had not changed state.
                  Decode is flat on both axes, which is the point: the gate does not trade anything
                  away. It is the largest single confirmed effect in the survey so far.
