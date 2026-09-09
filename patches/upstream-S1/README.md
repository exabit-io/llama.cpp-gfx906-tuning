# S1 upstream candidate: the gfx906 MMQ config table

One patch against upstream `master` 5d806aa25 (2026-09-08), branch `s1-upstream` in the Exabit clone
(`/root/exabit-llama.cpp-s1`). It is the fork's tile-table commit f48d3790261e (author Marko Tombak, ML-gfx906
fork) with the brace the cherry-pick lost restored and the config applied from J = 32 up (v2, tag `s1-v2`; v1 applied it from J = 8 and read −7.5% at 16 decode rows on master, tag `s1-v1`), and nothing else: no repack, no compiler-flag change, no MMVQ
changes (those are the separate S1 MMVQ series, see NEXT-STEPS S1).

Contents: `mmq-config-gfx906.cuh` (Q8_0: 8 warps, J up to 128, otherwise the rdna2 table) and three hunks in
`mmq.cuh` (host selection on `GGML_CUDA_CC_VEGA20`, device selection on `__gfx906__`, the `J > 64` occupancy gate
in `mul_mat_q_switch_J`: never for MoE ids, never when the tile grid has fewer tiles than CUs).

Evidence: `reports/2026-09-08-tile-table-ablation.md` (key `tile_table_ablation`): on upstream b10288 with only this
change, Q8_0 pp2048 844.6 -> 1100.7 tok/s tp4 (+30%), 230.3 -> 315.4 one die (+37%), 89% of the fork's Q8_0 prefill
gain; tg128 and batches 1-8 unchanged, batched 16-32 slots +16-21%.

Measured on master 5d806aa25 (`reports/2026-09-08-review-followup.md` §6): pp2048 +29% tp4 / +34% one die, 32 rows +15%,
1/8/16 rows and tg128 unchanged, perplexity 5.6216 identical, `test-backend-ops -o MUL_MAT` 1288/1288 (build
`/opt/llama.cpp-s1-upstream-v2`, raw `data/raw/2026-09-08/qwen38-27b-s1v2*.md`). Before submitting: (1) nothing technical remains on this box; (2) the authorship is Marko Tombak's — agree with him whether he submits it, we submit it
with his authorship (as the patch is written), or it goes in as a joint change; do not send it without that.
