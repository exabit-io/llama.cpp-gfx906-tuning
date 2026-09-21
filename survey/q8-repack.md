patch:            q8-repack | mx-llama.cpp q8_repack/ (26 entangled commits + 2 separable) | Q8_0 weight repacking with --no-repack / -nr to disable | repacked narrow-batch mat-vec paths
axis:             multi-user
zero point:       build-substrate-v041 (ba82ea19a) measured 2026-09-20, --no-repack arm
recipe:           4 slots x 64K | q8_0 K and V | 125 W/die | --cache-ram 49152 | -ngl all | GGML_ENABLE_CUSTOM_AR=1, gate absent from this build
metric:           decode tok/s/slot at the 4x64K design point; improves requires q<0.10 and effect >= +2%
result:           multi 4x64K n=4: ON 15.432 +/- 0.024 / OFF 14.640 +/- 0.023 tok/s/slot; single 1x32K n=4 median: ON 39.685 / OFF 38.500 tok/s
effect:           decode +5.41% multi-user, +3.08% single-user; prefill +5.24% multi, +2.79% single
stats:            multi decode p=0.0286 q=0.0761 | multi prefill p=0.0286 q=0.0761 | single decode p=0.0571 q=0.0761 | single prefill p=0.0571 q=0.0761 — all PASS at q<0.10, BH across m=8 tests | n=4 fresh per arm per axis
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
