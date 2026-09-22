patch:            q4v-cache | build option GGML_CUDA_FA_QUANTS must include q8_0-q4_0 | runtime -ctv q4_0 | 4-bit value cache with 8-bit keys
axis:             single-user
zero point:       build-faq with -ctv q8_0 measured 2026-09-22 (same binary, one flag differs)
recipe:           1 x 254K PRIMARY (same total KV as the 4x64K multi-user point), 1 x 64K control | -ctk q8_0 | 125 W/die | --cache-ram 49152 | -ngl all | RCCL + gated custom AR
metric:           single-stream decode tok/s (prefill secondary), MEDIAN of n>=4 — the four-die single-stream stall fires about 1 run in 16 and shifts a mean by ~1.5%, most of the 2% floor; improves requires q<0.10 and effect >= +2%
result:           decode median 18.588 -> 20.132 tok/s (n=4)
effect:           decode +8.30%; prefill -0.09% (n.s.)
stats:            decode p=0.0571 q=0.0762 PASS | BH over m=8 | n=4 fresh per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          improves
bin:              both
would change if:  a longer-context or higher-slot cell shows the dequant cost overtaking the bandwidth saved, or a KL-divergence check (stricter than perplexity) finds a loss the 16K/6 ppl cannot see
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

CLOSED 2026-09-22:  Quality gate PASSED. PPL 5.6216 +/- 0.0624 (q4_0-V) against 5.6219 +/- 0.0626
                  (q8_0-V) on the same binary with one flag changed, 16K/6 chunks: a 0.0003 difference
                  against +/-0.0625 sampling error, both on the 5.62 reference cluster for v0.4.1-era
                  upstream. Quality-neutral.
                  So: +4.44% decode at 4x64K, +8.30% at 1x254K, prefill untouched, 23.5% less KV, no
                  measurable quality cost. Bin BOTH -- it improves the primary metric on both axes.
                  Operationally this is `-ctv q4_0` with `-ctk q8_0` on a build whose
                  GGML_CUDA_FA_QUANTS includes q8_0-q4_0. Without that kernel the mode still RUNS but
                  converts K and V to f16 and WARNS in the log (measured: q5_1, uncompiled, ran 8% slow at
                  identical prefill), which is how it came to be recorded as a loser.
                  OWED: optimize.py Q4V_SLOPE=0.092 has the wrong sign and still feeds the MILP; the
                  README KV-cache row says "18% slower and free in quality".

AXIS NOTE:        companion record to q4v-cache.md. R2.7 (corrected 2026-09-21) requires the single-user
                  axis to be measured at its OWN design point: 1 x 254K primary, 1 x 64K control.
                  Any single-user figure in this campaign dated before that correction was taken at
                  1 x 32K and is superseded -- a floor is not a design point.

SCOPED 2026-09-22 (lead):  This verdict is for **Qwen3.8-27B-Q8_0 only**, and the record did not say so.
                  The KV cache is 31.5% of the bytes moved per decode step on this model at 4x64K (16 of
                  65 blocks hold attention, 4 KV heads, head_dim 256). On Qwen3.8-Flash-Next -- a
                  supported model under R3.10 -- the same arithmetic gives 12.8 KiB/token against 103.7
                  GiB of weights, so KV is **3.1%** of per-step bytes. The same 23.5% KV saving buys
                  almost nothing there and the dequant cost could make it a net loss.
                  A "bin: both" reads as universal. It is not: it is one model at three shapes. Flash-Next
                  needs its own sweep before any KV type is chosen for it, and the choice belongs per
                  model at launch (optimize.py), not as a global default in launch.sh.
