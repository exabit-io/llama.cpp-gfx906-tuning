# Promotion decision, 2026-09-22 — what ships

> **History (2026-09-22). Superseded:** the substrate is now `exabit-io/mx-llama.cpp` `merge-v0.5.0` on llama.cpp v0.5.0 (REQUIREMENTS R3.7) and the working plan is `RE-RE-SURVEY-ACTION-PLAN.md`. Numbers here were measured on the v0.4.1 base.


Decided by the technical lead on confirmed measurements (n=4 per arm, exact permutation, BH FDR
q<0.10, full-precision metrics, at the R2.7 design points 4x64K multi-user and 1x254K single-user,
125 W/die, `--cache-ram 49152`, `-ngl all`). Every item below is CONFIRMED, not screened.

## The promoted configuration

| element | setting | confirmed effect | evidence |
|---|---|---|---|
| base | v0.4.1 + mxxm substrate + our 20 terms | +17.4% decode, +25.0% prefill vs stock | E1 |
| collective | **`GGML_HIP_RCCL=ON`** | **+18.6% prefill** (4x64K), +11.3% (1x254K) | `survey/rccl-collective.md` |
| FA kernels | **`GGML_CUDA_FA_QUANTS=f16-f16;q8_0-q8_0;q8_0-q4_0`** | enables the V-cache win below; an uncompiled combo silently takes a slow path | `survey/q4v-cache.md` |
| custom AllReduce | **on, gated**: `GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481` | +6.9% / +5.0% / +9.2% decode | `survey/custom-allreduce-ungated.md` |
| V cache | **`-ctv q4_0`** with `-ctk q8_0` | +4.4% / +8.3% decode, 23.5% less KV, PPL 5.6216 vs 5.6219 | `survey/q4v-cache.md` |
| weight repack | on (default; never `--no-repack`) | +13.6% prefill (4x64K), +13.4% decode (1x254K) | `survey/q8-repack.md` |

## Measured result of the promoted configuration

| cell | decode tok/s/slot | aggregate | vs stock v0.4.1 |
|---|---:|---:|---|
| 4 x 64K | **16.59** | **66.3 tok/s** | +22.8% decode, +25.1% prefill |
| 1 x 254K | **20.13** | — | +18.1% decode, +15.7% prefill |

R3.1 (>= 12 tok/s per request): PASS at both design points.

## Rejected / closed, with reasons

| item | verdict |
|---|---|
| `GGML_CUDA_FORCE_MMQ=ON` | neutral-drop. -0.01% to +0.02%, all p >= 0.37. The heuristic already picks MMQ. |
| `GGML_HIP_NO_VMM=OFF` | not-measurable. Does not compile on ROCm (HIP has no CUDA VMM API). This is why the fork pins it ON. |
| `bf16-bf16`, `q4_0-q4_0` in FA_QUANTS | dropped. gfx906 has no native bf16; a 4-bit K cache is not wanted. |
| TurboQuant / `turbo3` / `tq3_0` | not in scope now. Candidate for a later round, measured at 1x254K where the KV term dominates. Needs a ~1,800-line port. |
| multi-node tensor parallelism | out of scope (lead). `NEXT-STEPS.md` S7. |

## Superseded claims — do not cite these

- "+33% prefill from the fork tile table" — attributed by inspection, never isolated.
- "+23.84% prefill from the AR size gate" — the gate prevents a collapse; it creates no prefill gain.
- "substrate prefill machinery is +21% over stock" — measured butterfly-vs-butterfly.
- "q8_0-K/q4_0-V is non-functional / a measured loser" — it was absent by build config, and is a WIN.
- `optimize.py` `Q4V_SLOPE = 0.092` — wrong sign; the measured value is a gain, not a cost.

## Still open

- Which of the 20 Exabit terms earn their place: the bundle adds only +2.79% decode over the substrate.
- Which substrate commits carry its +14.2% / +24.9%: 53 separable units + 12 feature groups unscreened.
- A V-cache type sweep (`q4_1`, `q5_0`, `q5_1`): does the trend continue below 4.5 bits per value?

## AMENDED 2026-09-23 — f16 KV, by the lead's direction

`settings/launch.sh` now sets `-ctk f16 -ctv f16` **explicitly** in `COMMON`.

It was already f16 in effect: **no profile ever set `-ctk`/`-ctv`**, so every profile has always run
llama.cpp's f16 default. The q8_0 KV appeared only in the benchmark cells — which means the campaign was
measuring a configuration production does not use. That is a real error in how the cells were specified,
not a change of mind.

Now it is also measured rather than inherited. Full 7x7 FA sweep, 2026-09-23, n=2 per cell:

| | 4x64K | 8x32K | 1x254K |
|---|---:|---:|---:|
| f16 vs q8_0/q8_0 decode | **+27.5%** | **+18.8%** | **+39.2%** |

And at 8 slots it decides requirement compliance, not just speed: f16 gives **14.13 tok/s** against R3.1's
floor of 12, while q8_0/q8_0 gives **11.89** and fails.

Every launch profile fits with f16, verified against the 31 GiB/die budget (R3.2) with 6.8 GiB/die of
weights:

    single 1x256K 10.8 | team/busy 16x32K 14.8 | pairs 8x64K 22.8 (2 dies) | long 2x256K 14.8
    long8 8x128K 22.8  | ceiling 8x160K 26.8   | ingest 12x32K 12.8

Only 8x192K does not fit (30.8 GiB/die before compute buffers) and no profile uses it.

**Standing caveat:** this rests on an n=2 screen, not an n=4 confirmation. The effect is large and
consistent in direction across three independent cells, and the alternative was already the default, so
the risk of acting now is low — but it is not a confirmed verdict and `survey-lint.py` will not let it be
recorded as one until it is.
