# Response to the 2026-09-08 review (technical lead, 2026-09-08 afternoon)

The review was written between 07:33 and 07:49 UTC, before the day's measurements (`reports/2026-09-08-next-steps-measurements.md`). Every claim was checked against the code and data on the box; this records what was applied, what the day's results had already settled, and what was declined.

## Applied

| Item | Verification | Action |
|---|---|---|
| P1 Q4V selection crashes (`n4_q8_0/q4_0` ladder key) | reproduced on the current optimizer (`KeyError`) | patch 0001 applied (offsets 16 lines after the day's edits) |
| P1 shell comment swallows the MTP flags in the generated command | confirmed by reading `fmt()` | patch 0001 applied |
| P1 topology/graph factors evaluated as tp4 for every placement | reviewer's tests fail on the original, pass after | patch 0002 applied; single-die rows now show `topo default` (were credited with the tp4 topology gain) |
| P2 production decode factors from the upstream-fastpath table (`_PB` unused) | confirmed (`f_build` used `_MR`) | production factors now from the production cells (`production_build.bench`, M2 24/32) as base-term ratios; offline dp4 estimate 279.7 → 268.6 (reviewer predicted 268.1) |
| P2 deep prefill gets the full short-context tile gain | confirmed (flat 1.33) | depth-aware gain: measured 1.33–1.36 to 32K (M2), the matmul-only component model beyond, scaled at 32K: 1.20 at 128K, 1.15 at 256K; 128K prefill 691 → 623, 256K 492 → 425, labelled model |
| P2 `--measured-only` filters before composition | confirmed | filter moved after the build/quant/cap composition for the core cell |
| Fleet HA figure 165 tok/s at 155 W | confirmed in the run-through fleet section | corrected to 150.6 (2 × 75.3 at the 125 W cap) in the report's md and html |
| S4: gfx906 attention already uses `v_dot2_f32_f16` | verified: `common.cuh:753` defines it for `__gfx906__`; `fattn-vec.cuh` and `fattn-tile.cuh` use it | S4 rewritten: tile table and split pricing first, arithmetic only if the ISA shows it unpacked |
| S6: the Q8 dot ignores the activation sum | verified: `vec_dot_q8_0_q8_1` reads `__low2half(bq8_1->ds)` only | S6 bullet corrected to a producer-side `need_sum` specialisation with a cache-variant caveat |
| S5: the speculative wrapper truncates after drafting | accepted from the review's reading of `common/speculative.cpp` | S5 carries the implementation note (enforce inside the MTP loop; zero-budget semantics; count MTP decode calls) |
| S1: rebase guards (shared `MMVQ_MAX_BATCH_SIZE` assert, GCN table shared with CDNA, MoE/ID path; MMQ table with its `J > 64` gate) | consistent with the patch contents | folded into S1 with the review's acceptance list; TODO 15 |
| Tile-table attribution is by inspection, not ablation | correct: the run-through compared whole builds | TODO 12 (ablation) |
| Cap curve mixes request and decode throughput | correct | documented in README s.4; TODO 13 (phase-separated cap sweep) |
| Paired-run confidence intervals for the production table | reasonable | TODO 14 |
| "One clamp episode remains unexplained" | correct | README power row now says the supply explanation is the best-supported diagnosis |
| Profiler: count graph launches with the HIP runtime trace, union intervals per die, exclude warm-up | done that way in M1 (`tools/m1-analyze.py`) | — |

## Already settled by the day's measurements (the review asked for them)

- **Instrument graph and collective eligibility before replacing communication** — done: the fork's custom peer-write allreduce and whole-token graph were found, switched on and measured (+14% single stream; gate found at four rows; adopted in `settings/gfx906.env`). The review's caution that RCCL time was "a residual from a weight-streaming model" was right: the M1 trace measured 3.4 ms of RCCL kernels and the win came from the removed host round trips and the graph, not from a faster kernel.
- **Recover the MMVQ patches** — they were in `patches/` under other names than the README listed; fixed that morning. The production manifest (build flags, source tree, install prefixes) is in `patches/README.md` and `CLAUDE.md`.
- **Trace single-stream tp4 and one die; select work by critical-path time** — M1 (`reports/2026-09-08-m1-kernel-trace.md`).
- **Run the open production depth, 24/32-slot and MTP cells** — M2 and M3.
- **Do not budget the fitted per-layer residual as removable launch time** — confirmed the hard way: three fusions returned ~3%, multi-stream overlap 0.

## Declined or not adopted

- **Compile-time `need_sum` and a distinct cache variant now** — the saving is a few hundred nanoseconds per quantize launch; left on S6 as a small item.
- **Rerun the production table before anything else** — the day's interleaved `-r 3` runs (production vs the new build, five rounds) reproduce the s.11 cells within 1%; the paired CIs are queued as TODO 14 rather than blocking.
- **TheRock rocBLAS note** — informational; the fallback BLAS prefill was measured as a tie and retired, so no packaging work follows from it.

## Verification of the applied optimizer changes

`python3 review/2026-09-08/test_optimizer_patch.py optimize/optimize.py` → 6 tests OK on the patched file (2 failures + 3 errors before). `optimize/results.md` regenerated with all five commands plus `--measured-only`; winners unchanged, figures as above.
