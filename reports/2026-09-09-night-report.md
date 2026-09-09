# Night report, 2026-09-09: the fork-survey items measured at 32K on the four dies

Written for the morning. The survey (`reports/2026-09-09-fork-survey.md`) named nine things to take from the community forks. Six of them were built on the `gfx906` branch in the night and measured at 32K depth, which is the floor you set that afternoon; the box was yours to use and no one else touched it. Raw tables: `data/raw/2026-09-09/qwen38-27b-night0909*.md` (transcribed from `/root/rocm-tests/bench`), the chain scripts in `tools/night-0909-*.sh`. The builds are still installed under `/opt/llama.cpp-n*`.

## Summary

| Item (survey #) | Built as | Verdict at 32K | Numbers |
|---|---|---|---|
| MTP `out_ids` fix (1) | n1 | **kept**, correct by construction; the acceptance collapse does not reproduce in our configuration (no unified KV cache) | two concurrent 32K slots on the buggy and the fixed build: acceptance 0.72 / 0.67 on both, 46.3 tok/s aggregate on both |
| alex4300's head-256 tile rows (2) | n2 | **dropped as imported, replaced by the box's own row**: the imported single-stream row −7.7% on one die at 32K, the 4–32 rows −5% on batched decode at depth and +0.4% perplexity; the sweep found a 64-thread row that is +2.3% (s.8) | rocm0 tg128@32K 19.6 → 18.05 (imported) → 20.05 (sweep winner); batched 16 slots @32K 139.5 → 132.8 (imported rows) |
| DPP warp reductions (4) | n3 / n3b | **kept outside the attention kernels**: +2% four-die decode; inside the FA kernels they cost 3.6% of prefill at depth and 8% of the draft-3 verify path, outside they cost nothing (s.8) | tp4 tg128 54.5 → 55.6; rocm0 pp512@32K 259.5 (inside) vs 269.5 (outside); draft 3 at 32K 60.3 vs 65.6 |
| Adaptive MTP depth, PR 27210 (3) | n4 | **no overhead, no gain on this prompt**: the controller sits at the floor (identical accepted/drafted counts to fixed depth 3) | n4 draft 3 60.4 / 60.0, adaptive 3..10 58.8 / 59.3, adaptive 2..6 57.0 / 57.5 (prod draft 3: 72.7–74.3) |
| Upstream 2026-09-09 + mx-llama.cpp master merge (7, 8) | n5 | **clean**: 16141/16141 backend tests, parity with n3, perplexity identical; `build.sh` now compiles the FA kernels for the three KV pairs the profiles use (`GGML_CUDA_FA_QUANTS`) | tp4 pp2048 1358 / tg128 54.5 / @32K 815 / 52.8; ppl 5.6448 (= n3) |
| **Serving gate at 32K** (the survey's s.5) | prod vs n3 | **the branch passes by a wide margin**: at 32K prompts the server is prefill-dominated and the repack's prefill gain decides it | total tok/s at 8 / 16 clients: prod 812, 810 / 715, 718; branch 917, 941 / 791, 814 (+13–16% / +11–13%); TTFT 84–89 → 69–76 s and 48 → 42 s |
| **MTP gate at 32K** | prod vs r2 | still failed by the branch, same size as before | drafting off 53.3 vs 50.0 (−6%); draft 3 74 vs 66.7 (−10%); acceptance 0.68 on both |

Also found on the way: **without the S1b a3 kernel the branch loses 13% at 16 slots × 32K** (batched 139.5 vs production 160.0), so a3 is merged into the final candidate (s.8).

What changes the plan: **the promotion decision flips at 32K.** At 2K the branch lost the serving gate by 3–4% and the survey's plan was to fix the one-token kernel first. At 32K the serving gate is won by 11–16% on total throughput and 15% on time to first token, because a 32K request is 32K of prefill against 256 of decode; only the single-user MTP gate is still lost (−10%). The branch (with the DPP reductions, without the imported tile rows, on the merged base) is therefore promotable for the multi-user profiles now, with production kept for the single-user MTP profile until candidate 2 (the one-token kernel) lands.

## 1. What was built

On `night-0909` (from `gfx906` at eb39d6c7a): d0b7ef75f the `out_ids` fix (qwen35 and qwen35moe), fe3102bdd the GCN head-256 tile rows (alex4300's values, host-selected by `GGML_CUDA_CC_IS_GCN`, device-selected under `GCN`), 3771c572c the DPP reductions (furnace 32f284244 unchanged), 17dfa2336 + 7d3d582b6 the adaptive-MTP PR (its delta-net `t_min` hunk dropped on milpster's finding; a brace lost in the keep-both merge restored). `night-merge` = that + upstream master 2026-09-09 (13 commits; one conflict in `mmq.cu` where upstream added `ncols_opt` to `mmq_args`, resolved by keeping the fork's column-chunk loop and passing `ncols_max` for both initialisers) + mx-llama.cpp master (5 commits, clean) + the `build.sh` FA-quant list. `night-0909-n3b` = the DPP commit with `GGML_GCN_NO_DPP` defined at the top of `fattn-tile.cuh` and `fattn-vec.cuh`. `fa-sweep` = the tile rows as `GFX906_FA2..FA32` macros for the sweep.

Builds: r2 (`/opt/llama.cpp-gfx906-master-r2`, the branch's kernels before the night), n1..n5 as above, prod (`/opt/llama.cpp-prod`). Branch builds run with the repack on and `settings/gfx906.env`; production with `gfx906.env`. Every build's `libggml-hip.so` hash is in the raw file's header.

## 2. Kernel tests and perplexity

- n2 `test-backend-ops -o FLASH_ATTN_EXT` 2959/2959; n3 all ops 15128/15128; n5 all ops 16141/16141.
- Perplexity 16K/6 (tp4): r2 5.6173, n2 5.6418, n3 5.6448, n5 5.6448. The tile rows move it by +0.44% (fp16 accumulation order in the attention tile changes with the tile shape); the DPP reductions add +0.05%. Both are below the run-to-run band of the metric (±0.063) but the tile shift is the size of the upstream fidelity moves we tracked, so a tile row is not free numerically and must be checked when the sweep picks one.

## 3. llama-bench at depth 0 and 32K (two rounds, order rotated; both rounds agree to ±0.5%)

| build | tp4 pp2048 | tp4 tg128 | tp4 pp2048 @32K | tp4 tg128 @32K | rocm0 pp512 | rocm0 tg128 | rocm0 pp512 @32K | rocm0 tg128 @32K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r2 | 1361 / 1357 | 54.5 / 54.5 | 868 / 854 | 52.9 / 53.0 | 424.5 / 424.0 | 22.55 / 22.50 | 273.7 / 273.9 | 19.50 / 19.47 |
| n1 (out_ids) | 1359 / 1357 | 54.8 / 54.4 | 861 / 866 | 53.1 / 52.8 | 424.3 / 423.9 | 22.48 / 22.54 | 273.8 / 273.4 | 19.46 / 19.50 |
| n2 (+ tile rows) | 1360 / 1361 | 54.1 / 55.0 | 884 / 879 | 52.6 / 52.7 | 424.3 / 424.8 | 22.53 / 22.56 | **269.6 / 269.8** | **17.98 / 17.98** |
| n3 (+ DPP) | 1359 / 1356 | **55.6 / 55.6** | 862 / 860 | 53.2 / 53.2 | 424.4 / 424.6 | 22.72 / 22.69 | **259.6 / 260.0** | 18.07 / 18.06 |

n1 is kernel-identical to r2 and reads it back within 0.5%, which is the noise floor of this stage. The single-stream cells on four dies do not move: each die holds one KV head there and attention is a small share of the token. On one die at 32K the imported single-stream row (`ncols = 2`: 128 threads, occupancy 8, 64-row KV tiles) loses 7.7% of decode against the RDNA2 row (256 threads, occupancy 2, 128-row tiles); alex4300 measured the opposite on a 60-CU MI50 at 16K under a 225 W cap with Q4_0 weights, so the row is theirs, not ours. The DPP reductions give the four-die single stream +2% at depth 0 and nothing at depth; on one die they cost the attention-heavy prefill at depth 3.6%, which is why the variant that keeps the generic reductions inside the attention kernels was built (s.8).

## 4. MTP gate at 32K (server, one slot, 32K prompt, 300 generated, wave 2 = decode only; two rounds each)

| build | drafting off | draft 3 | accepted / drafted |
|---|---:|---:|---|
| prod | 53.2 / 53.4 | 74.3 / 73.9 | 201 / 294 = 0.68 |
| r2 (branch, repack on) | 50.2 / 49.6 | 67.5 / 65.9 | 200 / 295 = 0.68 |
| n3 (+ tile rows + DPP) | 49.9 / 49.4 | 60.1 / 60.5 | 200 / 297 = 0.67 |
| n4 (+ adaptive MTP), draft 3 | 50.3 / 49.8 | 60.4 / 60.0 | 200 / 297 = 0.67 |
| n4 adaptive 3..10 | — | 58.8 / 59.3 | 200 / 297 = 0.67 |
| n4 adaptive 2..6 | — | 57.0 / 57.5 | 196 / 283 = 0.69 |

The gate is still lost by the branch (−6% plain, −10% with draft 3, identical acceptance: the verify-step cost of the repacked kernel, as in the follow-up report). The imported tile rows and/or DPP cost the draft-3 path a further 10% (67 → 60): the verify batch is 4 rows × GQA 2 = 8 columns, exactly the `ncols = 8` row alex4300 changed. The adaptive controller of PR 27210 never left its floor on this prompt (the counts are identical to fixed depth 3), so it cannot help where acceptance is 0.67; its author's gains are on code and recall workloads where acceptance runs at 0.75–1.0. It stays in the tree as an option, off by default.

## 5. out_ids

Two concurrent 32K clients on `-np 2` with draft 3: r2 (bug present) and n1 (fixed) both give 0.716 / 0.666 acceptance for the two requests and 46.3 tok/s aggregate. The collapse alex4300 bisected needs a condition we do not run (their reproduction used `-kvu`, the unified cache, with backend sampling). The fix stays: it removes a graph topology that depends on the output count, which upstream avoids for the same reason.

## 6. Server level at 32K prompts (16 slots, 8 and 16 clients, 256 generated, two rounds rotated)

| build | 8 clients: total tok/s | per-request decode | TTFT s | 16 clients: total tok/s | per-request decode | TTFT s |
|---|---:|---:|---:|---:|---:|---:|
| prod | 812 / 810 | 4.6 / 3.9 | 89.0 / 83.7 | 715 / 718 | 1.3 / 1.3 | 48.1 / 48.0 |
| n3 | 917 / 941 | 3.9 / 3.7 | 75.9 / 68.6 | 791 / 814 | 1.3 / 1.3 | 42.2 / 42.1 |

At 32K a request is 32K tokens of prompt and 256 of output, so the server's total throughput is prefill throughput and the per-request decode rate is what the slot gets between other slots' prompt reads (1.3 tok/s at 16 clients on both builds: the head-of-line regime the guide describes, which is what the two-pair layout and a prompt cache are for). The branch's repack prefill gain (+21% on the batched bench) shows through as +11–16% total and 15% shorter first-token latency. This is the serving gate, and at 32K the branch passes it.

## 7. The merged base (n5)

`night-merge` builds in three minutes on the cache, passes 16141/16141, and measures at parity with n3 on every cell (tp4 pp2048 1354 / 1358, tg128 54.1 / 54.5, @32K 800 / 815 and 49.6 (one after-load stall) / 52.8; one die identical); perplexity 5.6448 = n3. Upstream's `GGML_CUDA_FA_QUANTS` default does not include the q8_0-K / q4_0-V pair the capacity mode uses; `build.sh` now lists the three pairs the profiles use.

## 8. The tile-row sweep, batched decode at 32K, and the DPP scope (finished 19:27)

**Single-stream tile row (`ncols = 2`), one die, tg128 at 32K, rows 4–32 held at the generic AMD values, two rounds (±0.03 tok/s), perplexity 16K/6 per candidate.** Fields are threads, occupancy hint, KV rows per tile, K columns per tile.

| candidate | row | one die tg128 @32K | one die pp512 @32K | four dies tg128 @32K | perplexity |
|---|---|---:|---:|---:|---:|
| **v6** | **64, 8, 64, 64** | **20.05 / 20.04 / 20.05 / 20.06** | 274.0 | 53.2 | 5.6199 |
| v10 | 64, 4, 64, 64 | 20.07 / 20.03 | 274.3 | 53.3 | 5.6199 |
| v8 | 64, 8, 64, 128 | 19.93 / 19.93 | 274.2 | 53.3 | 5.6199 |
| v12 | 64, 8, 128, 64 | 19.65 / 19.67 | 274.2 | 51.6 | 5.6199 |
| v0 (generic AMD row) | 256, 2, 128, 64 | 19.61 / 19.61 | 274.1 | 52.9 | 5.6199 |
| v1 (alex4300) | 128, 8, 64, 64 | 18.05 / 18.05 | 274.3 | 52.9 | 5.6199 |
| v7 | 128, 4, 64, 128 | 17.65 / 17.67 | 274.1 | 53.3 | 5.6199 |
| v3 / v11 | 128, 4/8, 128, 64 | 17.27 / 17.24 | 274.0 | 53.1 | 5.6199 |
| v4 | 256, 2, 128, 128 | 16.26 / 16.26 | 274.2 | 52.3 | 5.6199 |
| v2, v5, v9 | 256-thread 64-row, 512-thread, occupancy 16 | do not build (tile constraints) | | | |

One 64-lane block per tile with 64-row tiles is the shape this kernel wants on Vega at depth: +2.3% single-stream decode on one die over the generic row, nothing on four dies (one KV head per die there), prefill and perplexity untouched (5.6199 = the branch's 5.6173 + 0.003 from the DPP reductions this sweep branch carries). Every 128- and 256-thread variant loses, and the occupancy field is inert (v6 = v10). This is the row the final candidate carries; rows 4–32 stay on the generic table, since alex4300's values for them cost the draft-3 verify batch and moved perplexity by +0.4% (s.2, s.4).

**Batched decode at 32K depth, four dies, each sequence with its own 32K prompt, 8 and 16 slots, two rounds:**

| build | 8 slots tg | 16 slots tg |
|---|---:|---:|
| prod | 141.8 / 141.8 | **160.0 / 159.9** |
| r2 (branch) | 141.9 / 142.7 | 139.5 / 139.7 |
| n2 (+ imported rows) | 134.1 / 134.2 | 132.8 / 132.9 |
| n3 (+ DPP) | 135.4 / 135.4 | 134.1 / 134.1 |

At 8 slots the branch and production tie; at 16 slots × 32K the branch without the S1b kernel loses 13% — the 9–16-column repack cliff the follow-up report found at 2K is still there at depth. The imported rows cost 5% at both slot counts (the `ncols = 8/16` rows). **Consequence: the final candidate merges the `s1b-a` branch head (the a3 form, 46010953d), which is what closes the 16-slot gap; the tag of the same name is the earlier spilling form and must not be used.**

**DPP scope (n3b = the DPP reductions with the attention kernels kept on the generic `ds_bpermute` cascades, two rounds):**

| cell | n3 (DPP everywhere) | n3b (DPP outside attention) |
|---|---:|---:|
| one die pp512 @32K | 259.4 / 259.5 | **269.2 / 269.5** |
| one die tg128 @32K | 18.03 / 18.04 | 18.04 / 18.05 |
| four dies pp2048 @32K | 863 / 851 | 892 / 899 |
| four dies tg128 | 55.6 / 55.1 | 55.4 / 54.7 |
| draft-3 gate at 32K (wave 2) | 60.3 / 60.8 | **65.5 / 65.7** |

The reductions inside the attention kernels were most of the verify-path loss (r2 read 66–67 on that cell before the night's changes) and all of the prefill-at-depth loss; outside attention they keep the +2% single-stream decode on four dies. The final candidate carries `GGML_GCN_NO_DPP` in `fattn-tile.cuh` and `fattn-vec.cuh`.

## 8b. The final candidate

`night-merge` = upstream master 2026-09-09 + mx-llama.cpp master + the `out_ids` fix + the DPP reductions (outside attention) + the measured `ncols = 2` row + adaptive MTP (off by default) + the S1b a3 kernel + the FA-quant build list, built as `/opt/llama.cpp-gfx906-20260909` and validated against production at 32K (`tools/night-0909-final.sh`): the table below is filled in from section 9 of the raw file when the run finishes.

| cell | production | final candidate | change |
|---|---:|---:|---:|
| `test-backend-ops`, all ops | — | 16141 / 16141 | clean |
| four dies pp2048 | 1132 / 1127 | 1354 / 1355 | **+20%** |
| four dies tg128 | 57.3 / 57.5 | 54.9 / 55.2 | −4% |
| four dies pp2048 @32K | 777 / 776 | 846 / 848 | **+9%** |
| four dies tg128 @32K | 54.2 / 54.4 | 53.4 / 53.5 | −1.5% |
| one die pp512 | 311 / 311 | 424 / 424 | **+36%** |
| one die tg128 | 20.7 / 20.6 | 22.6 / 22.6 | **+9%** |
| one die pp512 @32K | 214 / 214 | 274 / 274 | **+28%** |
| one die tg128 @32K | 18.06 / 18.04 | 20.07 / 20.09 | **+11%** |
| batched 8 slots × 32K (pp / tg) | 958 / 141.6 | 1131 / 141.5 | prefill +18%, decode parity |
| batched 16 slots × 32K (pp / tg) | 958 / 160.4 | 1134 / 159.4 | prefill +18%, decode parity |
| single user 32K, drafting off (wave 2) | 53.1 | 50.7 | −4.5% |
| single user 32K, draft 3 (wave 2) | 73.7 (acc 0.68) | 65.2 (acc 0.67) | **−11.5%** |
| perplexity 16K/6 | 5.5969 (its lineage) | 5.6099 | upstream's reference band (5.62), slightly below it |

Two rounds each where two numbers are shown; the batched and MTP cells are single rounds (the two-round versions of the same cells are in sections 4, 6 and 8 and agree). The candidate wins every prefill cell by 9–36%, wins one-die decode at every depth, ties four-die batched decode at 8 and 16 slots at 32K (the a3 kernel), and loses the four-die single stream by 4% plain and 11.5% with draft 3: the one-token repacked kernel, candidate 2's target, unchanged by the night.

**Promotion.** `/opt/llama.cpp-gfx906-20260909` (a symlink `/opt/llama.cpp-gfx906` points at it) is now the build behind the multi-user profiles in `settings/launch.sh` (`team`, `busy`, `pairs`, `ingest`, `batch`: variable `LLAMA_MULTI`); `/opt/llama.cpp-prod` stays behind the single-user MTP profiles (`single`, `long`, `long8`, `ceiling`: `LLAMA_PROD`) until candidate 2 lands. The `gfx906` branch of exabit-io/llama.cpp is fast-forwarded to this state and tagged `gfx906-20260909`.

## 9. What to do with it

1. Done in the night: the merged base + `out_ids` + DPP outside attention + the box's own single-stream tile row + adaptive MTP (off) + S1b a3 is `/opt/llama.cpp-gfx906-20260909`, promoted for the multi-user profiles; production stays for the single-user MTP profiles. `gfx906` on GitHub is this state.
2. Candidate 2 (one-token whole-block load on the two-plane layout) remains the item that closes the MTP gate (−11.5% at 32K draft 3) and the plain single stream (−4.5%); once it lands, one binary serves every profile.
3. Tile rows for 4–32 columns: a sweep on this box with the draft-3 verify batch and the 8/16-slot batched cells as the metric; the imported rows are out, the generic AMD rows stay until then.
4. Adaptive MTP: measure once on a code prompt (acceptance ≥ 0.75) before deciding a default; on prose it is inert. The out_ids collapse: try `-kvu` once to reproduce it.
5. The guide's gates are 32K gates from now on: `tools/night-0909-ab.sh` (stage 2 with own prompts per sequence, as in `night-0909-bb32k.sh`) and `night-0909-final.sh` are the templates; BENCHMARKS-TODO 16 and 17 are closed by this report.
6. The `s1b-a` name is both a tag (the first, spilling form) and a branch (the a3 form): always merge `refs/heads/s1b-a`.
