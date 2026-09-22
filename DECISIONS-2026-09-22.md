# Promotion decision, 2026-09-22 — what ships

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
