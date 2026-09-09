# The `gfx906` branch (upstream master 2026-09-08 merged into the fork tag b10912 + the Exabit series) against production — 2026-09-08 evening

`tools/gfx906-master-validate.sh` on `/opt/llama.cpp-gfx906-master` (branch commit 5b7794476 + docs; the same binary as `…-r2` except the MUL_MAT_ID fix), production = `/opt/llama.cpp-prod` (fork b10254 + the series), `settings/gfx906.env` on both. Source: `qwen38-27b-gfx906-master-validate.md`, `qwen38-27b-post22.md`, `bisect/ppl.md` in the bench folder.

## A. test-backend-ops (ROCm0)
MUL_MAT 1288/1288; RMS_NORM / GATED_DELTA_NET / L2_NORM / ADD 51/51; **MUL_MAT_ID asserted** (`ggml_cuda_mul_mat_id_needs_sync` tested the 16-wide dense MMVQ constant while MUL_MAT_ID keeps the 8-wide window; a 9–16-token f16/bf16 expert product on AMD was predicted "no sync" and fell through). Pristine upstream and pristine fork pass 880/880. Fixed in the branch (commit 5d36d6fc8), rebuilt as `/opt/llama.cpp-gfx906-master-r2`: MUL_MAT_ID 880/880, MUL_MAT 1288/1288. The production lineage (b10254/b10288) predates the predicate and is unaffected.

## B. perplexity 16K (6 chunks, tp4)
| build | ppl |
|---|---:|
| production (fork b10254 + series) | 5.5969 |
| gfx906 branch | 5.6173 ± 0.0624 |
| pristine upstream master 5d806aa25 | 5.6216 ± 0.0625 |

The 0.4% shift is upstream's, in two steps (key `upstream_ppl_bisect`, pristine builds at nine first-parent commits): commits 1–68 (2026-09-02 → 09-06) all read 5.6118, commits 69–104 read 5.6216. Step 2 (+0.17%) is 5fdfa6282, "models : fix GDN normalization from max to rsqrt" — the q/k L2 norm now carries eps inside the root as the flash-linear-attention reference does (its author expected no practical change; on this model it moves the loss). Step 1 (+0.27%) is before the fork base: 0f3a71be1 (09-02) itself reads 5.6118, so it lies among the 472 commits between stock b10288 (2026-08-05) and 09-02 (seven midpoints built; perplexities queued after the FA-counter job). The pristine fork tag b10912 reads 5.6143 and the branch 5.6173: the fork's kernels sit 0.004 under upstream at each step. Both are upstream fidelity changes, not our kernels: the branch with the repack kernels reads 5.6173, 0.08% under pristine master. **5.62 is the reference perplexity from here on.**

## C. decode numerics against production (64-token batches, 16K tokens)
Greedy text diverges at character 199; mean KLD 0.0036 ± 0.0012, 90th percentile 0.0012, same top token 98.9%. Consistent with the upstream change in B (the 2026-09-08 series alone measured 1e-5 against production on the same base).

## D. llama-bench -p 2048 -n 128 -r 3
| build | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |
|---|---:|---:|---:|---:|
| production (b10254 + series) | 1126.2 ± 1.2 | 58.0 ± 1.8 | 331.3 ± 0.1 | 21.74 ± 0.04 |
| gfx906 branch (repack on, default) | **1360.6 ± 3.1** | 42.8 ± 13.8 | **424.0 ± 0.1** | 20.4 ± 1.8 |
| fork b10912 pristine (repack on) | 1364.3 ± 2.8 | 41.3 ± 7.8 | 424.9 ± 0.2 | 21.5 ± 0.7 |
| upstream master 5d806aa25 pristine | 843.3 ± 1.2 | 46.1 ± 2.2 | 233.9 ± 0.1 | 20.33 ± 0.02 |
| stock b10288 (paired CIs, for reference) | 844.0 ± 0.6 | 46.5 ± 0.5 | 233.8 ± 0.2 | 20.16 ± 0.05 |

Upstream by itself has not moved on gfx906 since b10288 (2026-09-0x): every gain in the table is the fork's and the series'.

**The fork's b10912 state repacks Q8_0 weights at upload** (`q8_repack/`: int8 quant plane + f16 scale plane, its own dp4a mat-vec and tiled GEMM, on by default on gfx906, `--no-repack` / `-nr 1` off). Prefill +21% on four dies and +28% on one die over production — the same on the pristine fork, so it is the repack, not the merge or the series. Decode falls to 41–43 tok/s with a spread of ±8–14: the repacked path serves one-token products with its own mat-vec (`mul_mat_vec_q8_0_repacked`) and 2..N-token products with per-width variants or the tiled GEMM, so the series' 16-column MMVQ fast path (the +21% single stream, +62% at 12 slots) never runs on repacked weights. The no-repack row (post22) settles what the branch is worth today; the port of the fast path to the two-plane layout is the next kernel item (S1b).

## E. decode by width (llama-batched-bench at 2K, tok/s; repack on)
| slots | 1 | 2 | 3 | 4 | 8 | 9 | 12 | 16 | 17 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| production tp4 | 53.5 | 91.3 | 122.7 | 127.7 | 175.1 | 183.5 | 197.2 | 202.7 | 153.3 | 191.9 | 214.8 |
| gfx906 branch tp4 | **25.8** | **39.8** | 114.4 | 127.2 | 174.4 | **113.1** | **141.4** | **169.2** | 166.0 | **219.7** | **258.6** |
| production rocm0 | 20.6 | 37.6 | | 53.6 | 70.3 | | | | | | |
| gfx906 branch rocm0 | 22.4 | 37.6 | | 56.2 | 68.8 | | | | | | |

The repacked path is a different machine: one die is fine to slightly better (1 slot +9%, 4 slots +5%); on the split, 1–2 streams collapse (25.8 / 39.8 against 53.5 / 91.3: the series' single-stream work is bypassed and something else stalls — the four-die trace in post22 says what), 3–8 match production, 9–16 fall into the repacked path's 32-wide GEMM (141 vs 197 at 12 slots: the cliff the 16-column MMVQ removed), and 24–32 slots gain 15–20% (the repacked GEMM). llama-bench's ±8–14 spread at one stream on the split is the same collapse.

## F. single user with MTP draft 3 on the branch (repack on; llama-server -np 1, decode-only wave)
| `LLAMA_ENABLE_MTP_OPT` | 2K | 32K | acceptance |
|---|---:|---:|---|
| off | 76.0 | 65.8 | 0.72 / 0.60 |
| on | 76.3 | 66.4 | 0.72 / 0.68 |
| production (final-config, for reference) | 78.5 | 73.9 | |

The 4-row verify batches go through the repacked path's per-width mat-vec and land within 3% of production at 2K and 10% below at 32K; the fork's MTP flag changes nothing on this model.

## G. `--no-repack` on the branch (post22, r2 build)
| build | repack | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |
|---|---|---:|---:|---:|---:|
| gfx906 branch r2 | off | 1137.0 ± 0.9 | **37.9 ± 16.4** | 338.8 ± 0.02 | 21.43 ± 0.10 |
| gfx906 branch r2 | on (same session, run right after) | 1361.5 ± 2.2 | 53.7 ± 2.7 | 424.1 ± 0.03 | 21.90 ± 0.15 |
| production | (none) | 1128.4 ± 1.1 | 58.2 ± 1.8 | 331.7 ± 0.1 | 21.73 ± 0.05 |

Without the repack the branch reproduces production's prefill (+0.8% / +2%) and one-die decode (−1.4%), but the four-die single stream still collapses (37.9 ± 16.4). The repack-on row that followed in the same session read 53.7 ± 2.7 — the stall is intermittent, not a property of a build or a knob, which is why every earlier row carried ±8–16. So the single-stream stall is not the repack: it is in the fork's b10912 tensor-parallel state (the pristine fork shows it, pristine upstream does not, one die does not). A knob sweep (post22b: HIP graphs, the fork's whole-token graph, its concurrent lane dispatch, graph reuse, graph optimisation, hardware queues, the dedicated copy stream, the custom allreduce) isolates the feature; the four-die traces in post22 show where the time goes.

Decode by width with `--no-repack` (batched bench at 2K, tp4): 1: 31.3, 2: 59.3, 4: 129.4, 8: 158.5, 12: 196.5, 16: 201.6, 24: 194.1, 32: 212.8 — against production 53.5 / 91.3 / 127.7 / 175.1 / 197.2 / 202.7 / 191.9 / 214.8. From 12 slots up the branch without the repack is production; 1–2 streams carry the stall (−40%, −35%) and 8 slots read 9% low in this run.

## H. Knob sweep on the stall (post22b; r2, tp4, `--no-repack`, llama-bench tg128 × 3, gfx906.env base)
| variant | tg128 |
|---|---:|
| base | 37.9 ± 17.6 |
| HIP graphs off | 42.6 ± 15.7 |
| whole-token graph off (`GGML_META_TOKEN_GRAPH=0`) | 39.2 ± 15.3 |
| concurrent lane dispatch off (`GGML_META_PARALLEL_DISPATCH=0`) | 39.2 ± 15.1 |
| graph reuse off (`LLAMA_GRAPH_REUSE_DISABLE=1`) | 20.4 ± 5.6 |
| graph optimisation off (`GGML_CUDA_GRAPH_OPT=0`) | 38.6 ± 16.2 |
| `GPU_MAX_HW_QUEUES=8` | 38.0 ± 16.4 |
| dedicated copy stream off (`GGML_META_NO_DEDICATED_CPY=1`) | 39.9 ± 16.2 |
| custom allreduce off | 37.1 ± 13.2 |
| all fork paths off | 14.0 ± 2.9 |

No runtime knob removes the ±13–17 swing: the stall is code that is always on in the fork's b10912 tensor-parallel state, or an upstream change the fork merged (upstream pristine is stable at 46 ± 2 because it does not run the fork's meta backend). Next: per-sample timings (which of the three repeats stalls, and how) and a bisect over the fork's 126 first-parent commits between its two tags (`tools/bisect/fork-build.sh`, `tg-test.sh`; midpoints pre-built).

## I. Per-sample timing: the stall is a warm-up (post22c; tp4 tg128 × 5, per-repeat tok/s after llama-bench's own warm-up run)
| build | mean ± sd | samples |
|---|---:|---|
| gfx906 branch r2, repack on | 48.3 ± 12.1 | 41.5 30.0 **56.6 56.6 56.6** |
| gfx906 branch r2, repack off | 55.3 ± 1.8 | 52.8 53.8 **56.7 56.5 56.5** |
| production | 56.8 ± 1.6 | 53.9 **57.5 57.7 57.6 57.5** |

The branch's steady-state single stream is 56.5–56.6 tok/s, 1.8% under production's 57.6 — inside the ±2% band. What the three-repeat runs saw as "42 ± 14" is the first 100–300 tokens after model load running at 30–54 tok/s, a warm-up whose length varies from run to run (production has a shorter one: one sample at 53.9). Every earlier ± figure on the branch is this warm-up, not a stall in service. Whether it recurs in a server (after each prompt's batch-shape change) is the question the server-level check (post22d) answers; the fork bisect round says which fork commit lengthened it.

## J. Fork-history bisect of the warm-up (post22c/e; `tools/bisect/fork-build.sh`, `tg-test.sh`; per-sample tp4 tg128 × 5)
| fork position (first-parent b10254 → b10912) | date | samples | steady |
|---|---|---|---:|
| 16 · 803f00bb7 | 08-08 | 52.2 53.2 54.3 54.3 54.1 | 54.3 (no long warm-up) |
| 32 · 8253e3fbf | 08-18 | 50.7 51.8 54.3 54.3 54.0 | 54.3 (no long warm-up) |
| 48 · 115a32c1e | 08-22 | **30.4 17.6** 53.5 53.5 53.1 | 53.5 |
| 112 · 16628b027 | 09-03 | **23.1 14.7** 54.1 54.4 54.4 | 54.4 |
| b10912 pristine | 09-08 | 42.5 38.2 55.3 55.2 55.0 | 55.2 |
| gfx906 branch r2 | | 41.5 30.0 56.6 56.6 56.6 | 56.6 |
| production (b10254 + series) | | 53.9 57.5 57.7 57.6 57.5 | 57.6 |
| upstream master pristine | | 41.4 46.6 46.6 46.6 46.5 | 46.6 |

The long warm-up (two 128-token repeats at 15–40 tok/s after llama-bench's own warm-up run) enters the fork between positions 33 and 47 (2026-08-19 → 08-22), the fifteen commits that land the Q8_0 repack machinery (extra buffer types carrying the choice, the narrow-batch mat-vec, the row-interleaved layout, cached repacked views, composed buffer types scoped to the device). It persists with `--no-repack`, so it is the buffer-type or view path rather than the kernels. Round 2 tests every one of the fifteen (built; runs after the server-level check). The steady state improves along the fork's history (54.3 → 55.2) and the series adds its 2.5% on top (56.6).

## K. Server level (post22d; team profile, 16 slots, 1300/256 requests) — the promotion test
| build | 4 clients | 8 | 12 | 16 | TTFT s (4 → 16) |
|---|---:|---:|---:|---:|---|
| production | 67.4 | 76.1 | 83.0 | 83.4 | 4.98 → 5.77 |
| gfx906 branch, repack on | 62.0 (−8%) | 76.0 | 72.6 (−12.5%) | 79.6 (−4.6%) | 6.04 → 5.19 |
| gfx906 branch, `--no-repack` | 66.3 (−1.6%) | 73.5 (−3.4%) | 79.9 (−3.7%) | 81.2 (−2.6%) | 5.04 → 5.88 |

The after-load warm-up does not dominate a server (it is per process, not per request): with the repack off the branch sits 2–4% under production across the board, consistent with the 8-slot batched cell reading 9% low (158 vs 175) — a small regression in the ported 8-column MMVQ path to find alongside S1b. With the repack on, the 12-client row falls into the repacked GEMM's 9–16-row cliff and the 4-client row loses to its narrow-batch mat-vec.

## Verdict (2026-09-08 evening)
**Validated, not promoted.** Correct (one bug found and fixed; perplexity tracks upstream's own two fidelity fixes), faster where the fork's repack applies (prefill +21% / +28%, 24–32 slots +15–20%), Flash-Next runs on it — but at the server level it is 2–4% under production without the repack and 5–12% under with it, and the single-stream warm-up after load is two to three times longer. `/opt/llama.cpp-prod` stays on the 2026-09-08 build. The branch becomes production when (1) the fast path is ported to the repacked layout (S1b) so the 9–16-row cliff and the 4-client loss go away while the prefill and wide-batch gains stay, (2) the 8-column path is checked against the 2026-09-08 series (158 vs 175 at 8 slots), and (3) the warm-up's commit (bisect round 2, running) is understood — probably a buffer-type or cached-view path that also costs the first tokens of every server start.

Round 2 (positions 33–47 with the fork's defaults): position 33 (`e21ccb704`, "retire the enable env, extra buffer types carry the choice" — the commit that turned the Q8_0 repack **on by default**) shows the long warm-up (31.8, 24.0, then 54.1) where position 32 did not (50.7, 51.8, 54.3); position 34 (the commit that adds `-nr` to llama-bench) read 49.5, 47.5, 54.1 — a milder one. So the long warm-up tracks the repack being active more than a code change (round 1 and 2 ran with each build's default, which flipped at 33), and its length varies from run to run. The residual warm-up with `--no-repack` (first sample 52–54 against production's 54–57) is small. Mechanism not yet identified (a lazy first-use path in the repacked buffer type is the candidate: cached repacked views, first-touch repacking); it is per process, so a server pays it once at start. Left for the S1b work, where the repacked path is rewritten anyway.

Repeat runs (post22f): branch `--no-repack` 55.4 56.7 **58.5 58.3 58.2**; branch repack on 27.2 17.9 **57.1 56.9 56.8**; production 55.9 **59.1 59.1 59.0 58.9**; pristine fork `--no-repack` 52.3 53.1 55.4 55.3 55.1. Steady state of the branch without the repack is production's within 1–2%; the two-repeat warm-up is the repack's, every time. Key `branch_warmup_samples`.

## L. Promotion repeat (post23; interleaved, `--no-repack` branch vs production)
Batched decode at 2K, round 1: branch 163.6 / 192.3 / 201.3 at 8 / 12 / 16 slots, production 175.0 / 197.5 / 202.9 (−6.5% / −2.6% / −0.8%). The 8-column cell is a real, repeatable loss (158 and 164 against 175). The launch table at 8 columns is identical in both trees and upstream's DGX-Spark prefetch is compiled out for gfx906; the remaining suspect is the fork's mat-vec fusion routing (the gfx906 Q8_0 fast path requires an unfused product): post25 measures 8/12/16 with `GGML_CUDA_DISABLE_FUSION=1` on both builds. Round 2 and the server-level rows follow.

Complete repeat (two interleaved rounds; key `gfx906_branch_promotion_repeat`):

| | 8 slots | 12 | 16 | server 8 clients | server 16 clients |
|---|---:|---:|---:|---:|---:|
| production, rounds 1 / 2 | 175.0 / 174.8 | 197.5 / 197.5 | 202.9 / 202.9 | 77.3 / 77.9 | 81.8 / 82.1 |
| branch `--no-repack`, rounds 1 / 2 | 163.6 / 153.9 | 192.3 / 195.8 | 201.3 / 192.5 | 73.8 / 74.6 | 79.5 / 79.7 |

Production repeats to 0.1% (batched) and 1% (server); the branch is 6.5–12% under at 8 slots, 1–5% at 12–16 and 3–4.5% at the server level. The verdict stands.

## M. The first perplexity step (post23): between 2026-08-10 and 08-13
Pristine upstream at seven points between stock b10288 (08-05) and the fork base (09-02): 08-10 reads 5.5969 (the production figure), 08-13 and everything after 5.6118. Among the 59 commits between them the candidate is `e79e4bf66` (08-12), **"ggml-hip : remove -funsafe-math-optimizations"** — the HIP build lost `-fassociative-math` (IEEE-conformant floating point; upstream's reason was greedy argmax flipping on RDNA3.5 under MTP). The production lineage still builds with the flag. Two consequences to test: the perplexity pin (post26, the commit and its parent) and whether the flag is behind the branch's 3–4% server-level and 6–12% eight-column deficits (post27: the branch rebuilt with the flag, `/opt/llama.cpp-gfx906-master-r3`).

Pinned (post26): the parent `d86c7d62d` reads 5.5969, `e79e4bf66` reads 5.6118. **Step 1 is the compiler flag.** The production lineage's lower perplexity is fast-math reassociation, not a better kernel; upstream removed the flag for IEEE conformance (greedy argmax flipping on RDNA3.5 under MTP). Whether the flag is also worth the branch's 3–4% at the server level is post27.

r3 — the branch rebuilt with `-funsafe-math-optimizations` (verified in its compile flags) — reads **5.6212**, the same as without it. On today's code the flag no longer moves the numerics: the reassociation-sensitive path was the old `l2_norm` form of the GDN q/k norm, replaced upstream on 09-06 (step 2). Production's 5.5969 was the old norm form under fast-math; there is nothing to restore, and 5.62 is the reference. Whether the flag still buys speed is the rest of post27.

## N. The unsafe-math rebuild (post27, key `gfx906_branch_unsafe_math_r3`)
| | perplexity | single stream steady | 8 / 12 / 16 slots (two rounds) | server 8 / 16 clients |
|---|---:|---:|---|---|
| production | 5.5969 | 59.0 | 174.7 / 197.3 / 202.8 ; 174.6 / 197.1 / 202.9 | 77.2 / 81.8 |
| branch r2 (IEEE) | 5.6173 | 58.3 | 163.6 / 192.3 / 201.3 ; 153.9 / 195.8 / 192.5 | 73.8–74.6 / 79.5–79.7 |
| branch r3 (with `-funsafe-math-optimizations`) | 5.6212 | 59.0 | 151.7 / 189.4 / 196.0 ; 154.0 / 194.8 / 195.9 | 73.6 / 79.5 |

The flag is neither the numerics nor the speed on today's code. The 8-column loss (−12% at 8 slots) and the 3–4.5% server-level gap are in the branch's code. Remaining lead: the fork's mat-vec fusion routing around the gfx906 fast path (post25b, `GGML_CUDA_DISABLE_FUSION=1`). With the repack on, r3's width sweep repeats the validation's: 130 / 163 / 141 / 169 / 207 / 259 at 4 / 8 / 12 / 16 / 24 / 32 (the 9–16 cliff and the 24–32 gain).

## O. The eight-column loss (post25b, key `gfx906_branch_8col_check`)
| build | variant | -npl order | cells |
|---|---|---|---|
| branch r2 `--no-repack` | fusion on | 8, 12, 16 | 125.6 / 184.8 / 194.3 |
| branch r2 `--no-repack` | fusion off | 8, 12, 16 | 147.7 / 187.1 / 187.1 |
| branch r2 `--no-repack` | custom AR off | 8, 12, 16 | 153.3 / 187.7 / 193.9 |
| production | fusion on | 8, 12, 16 | 175.1 / 197.4 / 202.9 |
| production | fusion off | 8, 12, 16 | 169.2 / 190.8 / 196.2 |
| branch r2 `--no-repack` | fusion on | **16, 12, 8** | 187.3 / 194.8 / **142.7** |
| production | fusion on | 16, 12, 8 | 203.5 / 197.5 / 174.6 |
| branch r2, repack on | | 16, 12, 8 | 144.9 / 136.2 / 150.4 |

Not the fusion (production loses 3% without it; the branch gains nothing), not the custom allreduce, not the compiler flag (r3), not the order: reversed, the 8-slot cell is last and still reads 142.7 against production's 174.6, which is order-independent to 0.3%. A second, separate effect is the first cell after load on the branch (16 slots first: 187 vs 194–201 later) — the warm-up. The dense dispatch admits MMVQ to 16 columns on both trees and the 8-column launch table is identical, so the loss is in the merged kernel body at exactly eight columns (12 and 16 are within 1–3%). Left to S1b, which rewrites the narrow-batch path.

## Final verdict (2026-09-08, 20:20 UTC)
**Validated, not promoted; production stays on the 2026-09-08 build.** The branch is correct (one bug found and fixed; perplexity 5.617 against upstream's 5.622, both upstream fidelity changes), runs Flash-Next, and with the fork's repack is +21% / +28% on prefill and +15–20% at 24–32 slots — but it is 3–4.5% under production at the server level, 6–18% under at eight slots, has a 9–16-slot cliff with the repack on, and a longer warm-up after load. All of it sits in the fork's narrow-batch mat-vec paths and their interaction with the series; the port of the gfx906 Q8_0 fast path onto the repacked layout (NEXT-STEPS S1b) is the one piece of work that turns the branch into the production build with every gain kept. Acceptance for that promotion: the width sweep at or above production at every width 1–17 with the repack on, the team16 server within 2%, prefill at 1360, perplexity 5.62, `test-backend-ops` clean.
