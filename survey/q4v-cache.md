patch:            q4v-cache | build option GGML_CUDA_FA_QUANTS must include q8_0-q4_0 | runtime -ctv q4_0 | 4-bit value cache with 8-bit keys
axis:             multi-user
zero point:       build-faq with -ctv q8_0 measured 2026-09-22 (same binary, one flag differs)
recipe:           4x64K and 1x254K | -ctk q8_0, -ctv q8_0 vs q4_0 | 125 W/die | --cache-ram 49152 | -ngl all | RCCL + gated custom AR
metric:           decode tok/s/slot derived at full precision; improves requires q<0.10 and effect >= +2%; quality gate: PPL within sampling error of the q8_0-V arm
result:           4x64K 15.883 -> 16.587 | 1x254K median 18.588 -> 20.132 tok/s/slot (n=4 per arm per cell)
effect:           decode +4.44% multi-user, +8.30% single-user; prefill +/-0.09% (n.s.); KV 23.5% smaller (34.0 -> 26.0 KiB/token)
stats:            decode p=0.0286 q=0.0762 (4x64K), p=0.0571 q=0.0762 (1x254K) | prefill n.s. | BH over m=8 | n=4 fresh per arm per cell
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

RETRACTION 2026-09-22 (same day, before the ink dried):  My explanation for the historical figure was
                  WRONG and I propagated it into CLAUDE.md, README.md, optimize.py and a memory before
                  checking it.
                  What I claimed: the old "q4_0-V is slower" number timed an UNCOMPILED fallback path.
                  What is true: it was measured on /opt/llama.cpp-faq, an explicit
                  GGML_CUDA_FA_ALL_QUANTS=ON build (data/raw/2026-09-07/qwen38-27b-q8_0-faq.md). The
                  kernel was present. And the numbers were q8_0/q8_0 111.2 vs q8_0/q4_0 107.4 aggregate
                  decode at 8 sequences x 32K with ntg=128 -- a 3% cost against q8_0, not 18%. The 18%
                  was against f16.
                  So the discrepancy with today's result is NOT a build artefact. It is a difference of
                  SHAPE and PLATFORM: 8 slots / 32K / ntg=128 on a b10288-era base and ROCm 7.14, versus
                  4 slots / 64K and 1 slot / 254K with ntg=1024 on v0.4.1 + RCCL + ROCm 10.0. Which of
                  those dimensions flips the sign is UNKNOWN.
                  CONSEQUENCE FOR THIS VERDICT: it stands where measured -- 4 slots and 1 slot -- and is
                  NOT established at 8 slots, which is R2.2's other design point (8 x 192K). The old data
                  says q4_0-V loses 3% there. Until 8 slots is measured on the current build, this bin
                  should be read as "improves at 4 slots and single-stream", not unconditionally.
                  The fallback mechanism is real, but my EVIDENCE for it was bad: q5_1 (6.0 bits,
                  uncompiled) against q4_0 (4.5 bits, compiled) varies cache width AND compilation at
                  once, so the 8% gap is not attributable to the fallback. What establishes the
                  mechanism is the log line itself: "no FlashAttention vector kernel compiled for K/V
                  types q8_0-q5_1, converting K and V to f16 instead (slow)". It is not silent, and it
                  was not what happened in 2026-09-07. Detection is mechanical now:
                  tools/assert-no-fa-fallback.sh scans run logs for that line.
