# gfx906 patchset binning — action plan, third attempt (v0.5.0)

Written 2026-09-24 at the lead's instruction. **This file replaces `RESURVEY-ACTION-PLAN.md` as the working
plan.** The old file stays as history (method, traps, decisions); where they disagree, this file wins.
`llama.cpp-benchmarking/REQUIREMENTS.md` still outranks both.

## 1. Scope — the lead's words, not a paraphrase

> "your goal and your only goal is to bin llama.cpp patchsets"
> "you have a patchset x, you compile x, you test it, then you bin it"
> "we are building GGML_HIP_RCCL=ON and GGML_CUDA_FA_QUANTS=all and only using f16-f16, and no 8 slot
> bullshit, you do 4x64K for multi-user and 1x255K for single-user"
> "The only purpose in validating against Qwen 3.8 Flash-Next was to ensure that all the patchsets we are
> attempting to classify are compatible with qwen-4 architecture."

Not in scope, and not to be started without an explicit ask: KV-type tests of any kind, 8-slot or any other
cell, offload/config sweeps, Flash-Next performance numbers, new tooling, report rewrites.

## 2. Fixed test conditions — identical for every patchset, never varied

| item | value |
|---|---|
| base | llama.cpp **v0.5.0 = `7fe450e`** + **mxxm-t's fork** (`mxxm-t/mx-llama.cpp`, Marko Tombak) at its latest `eefc4e732` + the one llama.cpp patch from **mixa3607's ML-gfx906** (`mxxm-gfx906-kcase.patch`) — two different people and projects |
| build | `-DGGML_HIP_RCCL=ON -DGGML_CUDA_FA_QUANTS=all -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON`, gfx906, Release, ccache — same flags for every arm, checked by `assert-arms-comparable.sh` |
| model (perf) | Qwen3.8-27B Q8_0, four dies, `-sm tensor -ngl all -fa on` |
| KV | **f16 / f16** |
| multi-user cell | **4 x 64K** (`-npl 4 -npp 65536 -ntg 1024`) |
| single-user cell | **1 x 255K** (`-npl 1 -npp 260864 -ntg 1024`, context 262144 = `n_ctx_train`) |
| power | 125 W per die, host RAPL 150 W, clamp watchdog + SMC log |
| collective | RCCL (`GGML_CUDA_ALLREDUCE=nccl`, topology file) + custom AR gated (`GGML_TP_AR_MAX_NE=20481`) |
| compat gate | Qwen3.8-Flash-Next, production flags (`--n-cpu-moe 41`, mlock, `LLAMA_PLE_SHARD=1`), greedy 64 tokens: pass/fail only |

Measured cell times at f16: 4x64K = 6.9 min, 1x255K = 11.9 min. One arm, both axes, n=5 = 94 min.

## 3. The loop

For each patchset x:

1. **compile x** — the base with x added (our patches) or removed/switched off (substrate features).
   A patchset that does not apply or does not compile is recorded as that, with the error, and is not tested.
2. **test x** — both cells, n=5 runs per arm, interleaved with the base in seeded random order per block, so
   drift hits every arm alike. Axis-major: all multi-user blocks first, so multi-user bins land first.
3. **compat x** — the Flash-Next gate on x's build.
4. **bin x** — `binstats.py`, rule fixed before data: run-level mean (multi) / median (single); two-sided exact
   permutation, 5 vs 5; Benjamini-Hochberg q<0.10 over the round's declared family; an effect counts only
   if q<0.10 **and** |effect| >= 2%. Per axis: improves / regresses / neutral / mixed (mixed goes to the lead).
   Bin: improves on both -> `both`; one axis -> that profile only; regresses on both -> `regresses-both`;
   neither -> `neutral-drop`. A regression on one axis never blocks the other axis's build (R2.7).
5. **record + commit** — one `survey/<patchset>.md` record per patchset (lint-clean), the patchset committed
   onto its bin branch, pushed. Done per axis as results land, not at the end.

Why n=5 and not 4: n=4's exact-test floor (p=0.0286) cannot clear BH q<0.10 over a 24-test family unless
>=7 tests are real effects, so a clean but isolated win would bin as neutral. n=5's floor (0.0079) needs 2.
Found on synthetic data 2026-09-24 before any GPU time was spent.

## 4. Round 0 — the rebase onto v0.5.0 (in progress)

Done 2026-09-24 00:30 UTC:

- **Substrate** `gfx906-substrate-v041` -> **`gfx906-substrate-v050`** by `rebase --onto 7fe450e v0.4.1`.
  182 upstream commits; 9 conflict hunks in 5 files, all in the squash commit; the 7 follow-up commits and
  the terms commit replayed clean. Resolutions (semantic merges, never side-picking per hunk — T9):
  - `ggml-cuda.cu` graph_optimize: upstream restructured it (general `add_alloc_deps`, top-k MoE fusion
    deps); the fork's whole change there was 2 lines — gate the MoE weighted-reduction dep behind
    `ggml_cuda_moe_weighted_reduction_enabled()` and pass `for_alloc_deps=true`. Took upstream's function
    and applied those 2 lines to the MoE match only, so upstream's other fusion deps stay active.
  - `qwen4exp.cpp` (Flash-Next): upstream reshaped the HC/PLE norm gammas to `[n_embd, hc]` +
    `TENSOR_ALLOW_RESHAPE` and its merged graph code (`grouped_norm`) requires that shape; the fork changed
    the load flags (`trunk_flags` / `flags`). Kept upstream's shapes, OR'd in the fork's flags.
  - `llama-model.cpp`: upstream's new HRM_TEXT mirror guard prepended to the fork's split rules (the fork
    routes DSV4 through its own path, so upstream's in-lambda DSV4 block stays out, as it was on v0.4.1).
  - `llama-context.{h,cpp}`: upstream's graph-reuse guard `gf_res_prev_active` and the fork's sequence-layout
    tracking are independent; kept both.
- **Our terms** `c4-series` -> **`c4-series-v050`**: all 21 commits replayed with no conflicts; patch-ids
  identical before and after (content unchanged, only SHAs).
- **`gfx906-both`** -> **`gfx906-both-v050`** = substrate + the AR size gate (term 06), clean.
- Not ported, deliberately: `gfx906-v041`, `gfx906-v041-full` (superseded by `c4-series`), the `backup/*`
  presurvey branches (history).

Remaining in round 0 (~1 GPU h):

1. Build the v0.5.0 substrate (running) — a clean rebase is not evidence it compiles.
2. `assert-build-config.sh`; `test-backend-ops` on all four dies; perplexity 16K/6 on the 27B vs the
   reference cluster (quality gate R3.5, not a perf result).
3. On pass: move the canonical names — `gfx906-required` / `-single` / `-multi` -> the v0.5.0 substrate,
   `gfx906-both` -> `gfx906-both-v050-cfg` (5502cedbd: the AR gate + a commit making `GGML_HIP_RCCL=ON` and `GGML_CUDA_FA_QUANTS=all` the branch's CMake defaults — upstream defaults RCCL OFF, so until then a plain build of gfx906-both silently lacked RCCL), `c4-series` -> `c4-series-v050`; old tips kept as
   `backup/<name>-v041-20260924`; push everything. Nothing is pushed before it compiles and passes.

## 4b. Round 0b — rebase onto mxxm-t's latest fork + mixa3607's ML-gfx906 patch (lead, 2026-09-24 02:00)

**Two separate upstreams.** mxxm-t (Marko Tombak) = `mxxm-t/mx-llama.cpp`, the llama.cpp fork with the gfx906
kernel work. mixa3607 = `mixa3607/ML-gfx906`, ROCm-for-gfx906 builds and Docker presets, no llama.cpp source;
its newest preset builds stock ggml-org b11026, its mxxm-t preset pins the fork's old b10254, and all current
presets compile with `-mllvm -amdgpu-sched-strategy=max-ilp` (not yet in our builds — binned as a build-flag
patchset). An earlier version of this section said "the latest mxxm, as ML-gfx906 builds it"; that was wrong.

The lead: the v0.5.0 rebase was meant to bring in the latest of `mixa3607/ML-gfx906`'s llama.cpp, not our
2026-09-09 fork snapshot. The v0.5.0 rebase above carried mxxm `0c81bd502`; mxxm master had moved **34
commits** to `eefc4e732` (2026-09-21). ML-gfx906 fetched in full (all 5 branches, 33 tags, no submodules;
master `e1aa948`, 2026-09-19): its llama.cpp builds use the mxxm fork plus exactly one patch,
`mxxm-gfx906-kcase.patch` (Q4_K/Q5_K/Q6_K MMQ configs), identical on every branch that has one.

- **`gfx906-substrate-v050-mxxm`** = `gfx906-substrate-v050` + the 34 mxxm commits cherry-picked **one by one**
  (so each stays separable for binning) + the kcase patch as its own commit (`e9e43f917`). 3 conflicts, all
  merged semantically: `llama-graph.cpp` (union of the swiglu-clamp arch lists: upstream's MAPLE/HY_V4 + the
  commit's DEEPSEEK41), `ggml-backend-meta.cpp` (upstream removed the multi-buffer abort; kept that and the
  commit's mirror-lane block), `llama-model.cpp` (upstream's HRM_TEXT guard + the commit's tied-output rule).
- **`c4-series-v050m`**: our 21 terms on it, no conflicts, patch-ids identical.
- **`gfx906-both-v050m`** = substrate + AR size gate + RCCL/FA_QUANTS defaults + **one new commit making
  `GGML_TP_AR_MAX_NE=20481` the default** (`ac179c379`). Reason: mxxm `41c46cedb` turned custom AR on by
  default, so with the variable unset the fork's own 262144 threshold applied — the configuration measured at
  -19% prefill at 4x64K. Measurement scripts export 20481 explicitly, so no measured number changes.

The 34 commits join round 2's classification: custom-AR default and two-shot tuning (`41c46cedb`,
`92607b5d1`, `19d784ad0`) fall under the already-binned custom AllReduce; `de27e7509`, `20af9a480`,
`62d4be47d`, `be8ff98aa`, `a355590d2`, `cf6a98f73`, `54702a718`, `27f755681` are 27B-visible candidates (checked
per group before round 2 is built); `afca10209`, `2e740434e`, `d9c6fc44d` and the kcase patch are qwen4exp /
K-quant (Flash-Next compat); the DeepSeek-V4.1 / DSpark / gemma commits are other models (group C).

## 5. Round 1 — our patchsets (6 arms + base, ~11 GPU h)

Built on `gfx906-both` (v0.5.0). Commits cherry-picked from `c4-series-v050` (term numbers as before):

| patchset | terms | measured against | note |
|---|---|---|---|
| mmvq-q8-fastpath | 01 05 09 15 | base | one unit: 01 does not compile without 05 (`q8_fast`); 15 needs 01's define and is its MUL_MAT_ID fix |
| norm-add-fusion | 02 03 | base | |
| gdn-producer-fold | 04 07 08 10 12 | norm-add-fusion | cannot apply without 02 03; binned on its increment |
| s1b-repacked-matvec | 17 18 19 | base | |
| fa-head256-rows | 22 27 | base | |
| dpp-warp-reductions | 23 28 | base | |

Already in the base, not re-tested: term 06 (AR size gate). Not patchsets for this instrument: terms 11 13 14
16 20 25 29 (docs, build script, inventory), 21 24 26 (MTP; bin `technique-requires-implementation`, R3.9).

Timing: multi-user 7 arms x 5 x 6.9 min = 4.0 h, single-user 7 x 5 x 11.9 = 6.9 h, compat ~15 min.
Scripts: `night-20260919/binrun.sh`, `binstats.py`, `fncompat.sh` (commit SHAs updated to `c4-series-v050`).

## 6. Round 2 — the substrate's patchsets (8 arms + base, ~14 GPU h)

The substrate is the fork `mxxm-t/mx-llama.cpp` @ `0c81bd502` (148 commits over b10760), squashed. Its
commits cannot be removed one by one (most were rewritten by later ones), so its patchsets are **feature
groups**, removed from the base either by the fork's own runtime switch (same binary, cleanest contrast) or
by reverting the feature's code. Classified from the 148 commit subjects and the switch code (static, 0 GPU):

**Already binned `both` — carried forward, not re-tested (lead, 2026-09-24):** `tp-ar-size-gate` (our term 06, gfx906-both's own commit), `custom-allreduce` and `q8-repack` (substrate code, inherited by gfx906-both from the substrate beneath it), `rccl-collective` (the `GGML_HIP_RCCL=ON` build flag, in every build). Records: `survey/tp-ar-size-gate.md`, `custom-allreduce-*.md`, `q8-repack*.md`, `rccl-collective*.md`.

**A. Measurable on the 27B — these are the round-2 arms**

| patchset | key commits | removal |
|---|---|---|
| q8_1-activation-cache | 775a8051f, cfea4a1f6, 0a694d8ad | `GGML_CUDA_Q8_1_CACHE=0` |
| meta-token-graph | 751b6114c, 8051f5f2a, a3ab67c89, 693375a1f | `GGML_META_TG_LIMIT=0` |
| meta-xfer-rccl | 04f89ab03 (part) | `GGML_META_XFER_RCCL=0` |
| alloc-layout-cache | 6d2012d8a, 42b3cfb63, 687ef0194, d5047d6aa | `GGML_GALLOC_LAYOUT_CACHE=0` |
| gfx906-mmq-config | 5c4505b5d, 2335544ab, 5344ceea4, f1684f76c | revert `mmq-config-gfx906.cuh` + its hooks |
| gdn-chunked-prefill | 55b275262, 1248e5e37, 4405b44cd, 0e3249e9d | revert `gated_delta_net_chunk.*` + dispatch |
| meta-concurrent-lanes | 5d9efc8ca | revert (switch to be confirmed) |
| sched-input-staging | 38e266b6a, c7069d868 | revert (switch to be confirmed) |

A revert that does not compile makes that patchset **structural** (`neutral-required-substrate`,
`structural: required-by:<what broke>`) — recorded, not tested.

**B. Flash-Next / MoE only — the 27B never executes them.** Top-k and MoE routing (efa821ce8, 01d1c54d9,
a36820a9d, b549a9862, 1b7d76ffc, e02d4fe6c, e72087dd5, 31e3d2c5e), fused MoE mat-vec (b8baef09f, 50d5fdeba,
481f684d3, 7e7685b54, ef7246cd9), qwen4exp support (51b31fc47, cc3c8129b, b3a817edf, e8bad4d39, 6ff101bae,
c32b57591). Per the lead, Flash-Next is a compatibility check, so these are recorded as **required for
qwen4exp** once the compat gate passes on the base; no perf test.

**C. Not ours to measure — recorded by inspection, 0 GPU.** Other models (DeepSeek-V4, DSpark, MiniMax,
MiMo2), MXFP4 (not a quant we run), MTP/speculative (R3.9, `technique-requires-implementation`),
pipeline-parallel-only scheduling (we run one stage), out-of-memory fallbacks (robustness, keep), BF16-in-F32
(`upstream-already-has-it`, PR 28846), docs and CI.

How a substrate bin lands: the substrate stays whole in `gfx906-required`. A patchset that regresses on a
profile is switched off in that profile's launch settings (runtime switch) or reverted on that profile's
branch (code revert). Winners need no action — they are already in.

The group commit lists above are from subjects; before round 2 is built, each group's file/hunk ownership is
checked against the v0.5.0 substrate (host only), and group A is final only after that check.

## 7. Timeline (UTC, 2026-09-24 onward; GPU hours only, host work runs between rounds)

| step | GPU h | ends |
|---|---:|---|
| Round 0: build, test-backend-ops, perplexity, compat on base | ~1 | ~02:00 |
| Round 1 builds (ccache) | 0 | ~02:15 |
| Round 1 multi-user bins | 4.0 | ~06:15 |
| Round 1 single-user bins + compat | 7.1 | ~13:30 |
| Round 2 group check + builds (host) | 0 | ~14:30 |
| Round 2 multi-user bins | 5.2 | ~20:00 |
| Round 2 single-user bins + compat | 9.2 | 09-25 ~05:30 |

Total ~27 GPU h. If a phase overruns its time by more than 20%, stop and report instead of spending the next
phase's budget. Every launch gets a PID-based waiter and a monitor in the same turn.

## 8. What went wrong in attempts 1 and 2 — do not repeat

1. **Drift off the goal.** ~69 h of wall clock went mostly to KV sweeps, offload x slots sweeps and config
   studies while the bin branches stayed empty (`gfx906-both` 1 commit, `-single`/`-multi` 0).
2. **Refusing the goal.** After the lead said the goal was binning, the session killed its own binning run as
   "unasked" and wrote a handover advising against it. Binning is authorised by the goal itself.
3. **Measuring before checking the arms.** Mismatched FA_QUANTS, the wrong baseline, the AR size gate not
   compiled into the base (`build-substrate-allquants` lacks `GGML_TP_AR_MAX_NE`), 2K cells quoted as results.
   `binrun.sh` now refuses to measure unless every arm passes `assert-build-config.sh`,
   `assert-arms-comparable.sh` and has the AR gate compiled in.
4. **Plans that could not conclude.** n=3/n=4 designs whose exact-test floor cannot pass BH over the family.
   Check the design on synthetic data before spending GPU time.
5. **State in scratch space.** The 23:38 group screen died because its group list lived in the session's
   temporary directory. Everything a job needs lives next to the job.
6. **Clean merge taken as done.** A patch that applies is not a patch that compiles (term 01 needs 05's
   `q8_fast`). Build before claiming.

## 8a. Versioning — every state stays reachable after branches move

The bin branches (`gfx906-required/-both/-single/-multi`) are moving names: they are force-moved on each
rebase and gain commits each round. Every state they pass through is pinned by an **annotated tag** on
exabit-io/llama.cpp (immutable; the message says what the state is and what superseded it):

| namespace | pins | examples |
|---|---|---|
| `import/…` | fork snapshots we imported | `import/mxxm-0c81bd502` (the substrate source), `import/mxxm-b10912` |
| `gfx906/<base>/…` | each rebase's branch tips | `gfx906/v0.4.1/substrate`, `…/both`, `…/c4-series` (pushed 2026-09-24) |
| `gfx906/<base>/r<N>/…` | bin branches after round N's bins are committed | `gfx906/v0.5.0/r1/both`, `…/r1/single`, `…/r1/multi` |
| pre-v0.4.1 | older production states, kept as they were | `gfx906-20260909`, `gfx906-b11067-1d1361e`, `gfx906-pre-v041-20260920` |

Upstream's `v0.4.1` and `v0.5.0` tags are pushed too, so every base resolves inside our repo.
**GitHub releases** mark milestones only: a rebase that passed the round-0 gate (notes = gate results) and each
completed round (notes = the bins and their numbers). First release: `gfx906/v0.5.0` when the gate passes.

## 9. Where things are

| what | where |
|---|---|
| code repo | `/root/exabit-llama.cpp` (origin exabit-io/llama.cpp); worktrees `/root/wt-v050` (substrate), `/root/wt-c4-v050` (terms), `/root/wt-both-v050`, `/root/wt-bin` (arm builds) |
| run scripts | `/root/night-20260919/binrun.sh`, `binstats.py`, `fncompat.sh` (copies in `llama.cpp-benchmarking/tools/binrun/`) |
| records | `/root/llama.cpp-benchmarking/survey/*.md`, lint `survey/survey-lint.py` |
| requirements | `/root/llama.cpp-benchmarking/REQUIREMENTS.md` |
| history, traps T1-T12 | `/root/RESURVEY-ACTION-PLAN.md` Part 4 |
