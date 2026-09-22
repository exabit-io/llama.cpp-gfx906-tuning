# Raw measurement data, 2026-09-19 .. 2026-09-22 campaign

381 cell files and 14 TSV summaries from the v0.4.1 re-basing campaign. Every TSV column set is
`arm | cell | rep | decode | prefill` unless noted; decode and prefill in later files are DERIVED at
full precision by `tools/cell-metrics.py` (`n_tg/T_TG`, `n_pp*b/T_PP`) rather than read from the
tool's two-decimal rate columns — those produce ties that break a median permutation test.

| TSV | what it is | arms |
|---|---|---|
| `variance.tsv` | harness variance (Part 6 gate 3) | one build, repeated |
| `cmp.tsv` | substrate vs substrate+20 terms, first controlled A/B | R, Rours |
| `gate12.tsv` | Part 6 gates 1+2: stock zero point, single-user instrument | B, Bsu, Rsu, RoursSU |
| `stall.tsv` | four-die single-stream stall rate, n=12 per arm | Rsu, RoursSU |
| `d1.tsv` | D-1 runtime screen (custom AR, repack, AR size gate) | E1/E2/E3 |
| `d1conf.tsv`, `d1conf2.tsv` | D-5 confirmation, n=4, both axes | C-*, S-* |
| `d1ar.tsv`, `d1ar2.tsv` | does custom AR earn its place; reps 5-6 verify a method fix | M/S-AR-* |
| `rebase.tsv` | **48 cells, 4 arms on one binary via `GGML_CUDA_ALLREDUCE`** | rccl, rccl-customAR, butterfly, butterfly-customAR |
| `round1.tsv` | **72 cells: gate 1 redo, FORCE_MMQ, q4_0-V, repack on RCCL** | stock, substrate, bundle, mmq-*, v-q*, repack-* |
| `q4v.tsv` | perplexity gate for the q4_0 V cache | vq8, vq4 |
| `screen.tsv` | **DISCARD — delta-minus screen with mismatched FA_QUANTS** | see below |
| `next.tsv` | substrate rebuilt on the campaign config (incomplete, run stopped) | substrate-campaign |

## `screen.tsv` is invalid and is kept only as evidence

14 cells, ~1.6 GPU hours, wasted. The delta-minus builds carried the default
`GGML_CUDA_FA_QUANTS` while the comparison arm carried the campaign value, so each arm differed from
the bundle in TWO ways: the removed commit and the compiled attention-kernel set. All six
minus-builds came in 2.42%-2.69% below the bundle — identically. Six unrelated commits cannot each
cost the same 2.5%; that is the build difference, and it is above the 2% materiality floor.

`tools/assert-arms-comparable.sh` now refuses such a run in under a second, and it is a precondition
in the measurement scripts. The file is retained because the failure is more instructive than the
result would have been.

## Provenance

Scripts that produced all of this are in `tools/campaign-2026-09/`. Conditions unless stated: 125 W
per die, fans pinned, host RAPL 150 W, `-sm tensor` over four gfx906 dies, `-ngl all`, `-fa on`,
`-ctk q8_0`, `-b 2048 -ub 2048`, `--cache-ram 49152`, Qwen3.8-27B-Q8_0, ROCm 10.0.
Design points per REQUIREMENTS R2.7 as corrected 2026-09-21: **4 x 64K** multi-user, **1 x 254K**
single-user primary, **1 x 64K** control. 1 x 32K cells in the earlier files predate that correction
and are superseded — a floor is not a design point.
