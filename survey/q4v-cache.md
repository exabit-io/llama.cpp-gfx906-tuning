patch:            q4v-cache | build option GGML_CUDA_FA_QUANTS must include q8_0-q4_0 | runtime -ctv q4_0 | 4-bit value cache with 8-bit keys
axis:             multi-user
zero point:       build-faq with -ctv q8_0 measured 2026-09-22 (same binary, one flag differs)
recipe:           4x64K and 1x254K | -ctk q8_0, -ctv q8_0 vs q4_0 | 125 W/die | --cache-ram 49152 | -ngl all | RCCL + gated custom AR
metric:           decode tok/s/slot derived at full precision; improves requires q<0.10 and effect >= +2%
result:           4x64K 15.883 -> 16.587 | 1x254K median 18.588 -> 20.132 tok/s/slot (n=4 per arm per cell)
effect:           decode +4.44% multi-user, +8.30% single-user; prefill +/-0.09% (n.s.); KV 23.5% smaller (34.0 -> 26.0 KiB/token)
stats:            decode p=0.0286 q=0.0762 (4x64K), p=0.0571 q=0.0762 (1x254K) | prefill n.s. | BH over m=8 | n=4 fresh per arm per cell
evidence:         confirmed-fresh
structural:       standalone
verdict:          unresolved
bin:              unbinned-pending
would change if:  the perplexity gate shows a quality loss beyond the 0.04 KL budget the optimiser applies to quantisation choices
notes:            TWO THINGS WERE WRONG ABOUT THIS BEFORE TODAY, and they compounded.
                  1. It was absent by BUILD CONFIG. GGML_CUDA_FA_QUANTS is a string list and ours omitted
                     q8_0-q4_0 while compiling q4_0-q4_0 and bf16-bf16, neither of which is wanted here.
                     The fork's own recipe had it right. With the kernel missing, the mode looks
                     "non-functional", which is how CLAUDE.md recorded it.
                  2. Its measured cost had the SIGN WRONG. optimize.py carries Q4V_SLOPE=0.092 against
                     Q8_SLOPE=0.086 -- q4_0-V predicted 7% SLOWER. On the v0.4.1/RCCL build it is 4-8%
                     FASTER, which is what bandwidth-bound reasoning predicts once the dequant path is
                     the compiled one: fewer KV bytes read per step, and decode is bandwidth-bound here.
                  I predicted it would lose, on the strength of that stale coefficient, and said so before
                  measuring. The measurement disagreed in both cells.
                  VERDICT HELD PENDING QUALITY: a decode win from a lossier cache is only real if the loss
                  is acceptable. Perplexity gate running (q4v-quality.sh); optimize.py's Q4V_SLOPE and the
                  README KV-cache row must both be corrected once it lands.
