patch:            tp-ar-size-gate | exabit-io/llama.cpp terms/06-39e3ad8bb | GGML_TP_AR_MAX_NE explicit size gate | restrict the fork's custom AllReduce to tensors at or below N elements
axis:             multi-user
zero point:       build-c4series (substrate+20 terms, f04198b3f lineage) measured 2026-09-20, gate OFF arm
recipe:           4 slots x 64K | q8_0 K and V | 125 W/die | --cache-ram 49152 | -ngl all | env stated per arm, gfx906.env NOT sourced
metric:           prefill t/s at the 4x64K design point; improves requires q<0.10 and effect >= +2%
result:           screen n=2: gate ON 632.9 / gate OFF 510.8 t/s | fresh n=4 confirmation IN FLIGHT 2026-09-20 14:2x
effect:           +23.90% prefill, -0.21% decode (screen)
stats:            pending fresh confirmation; screen n=2 cannot produce a q value
evidence:         screened-only
structural:       standalone
verdict:          unresolved
bin:              unbinned-pending
would change if:  the fresh n=4 confirmation fails to reproduce the prefill gap, or the gap turns out to depend on --cache-ram rather than the gate
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
