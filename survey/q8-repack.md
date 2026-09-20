patch:            q8-repack | mx-llama.cpp q8_repack/ (26 entangled commits + 2 separable) | Q8_0 weight repacking with --no-repack / -nr to disable | repacked narrow-batch mat-vec paths
axis:             multi-user
zero point:       build-substrate-v041 (ba82ea19a) measured 2026-09-20, --no-repack arm
recipe:           4 slots x 64K | q8_0 K and V | 125 W/die | --cache-ram 49152 | -ngl all | GGML_ENABLE_CUSTOM_AR=1, gate absent from this build
metric:           decode tok/s/slot at the 4x64K design point; improves requires q<0.10 and effect >= +2%
result:           screen n=2: repack ON 15.471 / OFF 14.641 tok/s/slot | fresh n=4 confirmation IN FLIGHT
effect:           +5.67% decode, +4.56% prefill (screen)
stats:            pending fresh confirmation; screen n=2 cannot produce a q value
evidence:         screened-only
structural:       standalone
verdict:          unresolved
bin:              unbinned-pending
would change if:  the fresh confirmation does not reproduce it, or the single-user arm regresses (it would then be multi-user-only rather than both)
notes:            Runtime-switchable via --no-repack, so it needs no build surgery -- which is why it
                  was screened in the first round. Positive on BOTH metrics at the design point, so it
                  is the strongest 'both' candidate so far, pending the single-user arm now running.
                  26 of its commits are entangled (later commits rewrote the same lines), so if the
                  runtime switch ever disagrees with a build-level revert, the switch is the weaker
                  instrument: it may not disable every path the commits added.
