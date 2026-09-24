# gfx906 patchset binning — action plan (code: exabit-io/mx-llama.cpp, substrate = its master)

The working plan. `RESURVEY-ACTION-PLAN.md` is history (method, traps T1-T12, decisions); where the two
disagree, this file wins. `REQUIREMENTS.md` outranks both. Last restructured 2026-09-24 when the lead set the
substrate (section 2).

## 1. Scope — the lead's words, not a paraphrase

> "your goal and your only goal is to bin llama.cpp patchsets"
> "you have a patchset x, you compile x, you test it, then you bin it"
> "we are building GGML_HIP_RCCL=ON and GGML_CUDA_FA_QUANTS=all and only using f16-f16, and no 8 slot
> bullshit, you do 4x64K for multi-user and 1x255K for single-user"
> "The only purpose in validating against Qwen 3.8 Flash-Next was to ensure that all the patchsets we are
> attempting to classify are compatible with qwen-4 architecture."

Not in scope, and not to be started without an explicit ask: KV-type tests of any kind, 8-slot or any other
cell, offload/config sweeps, Flash-Next performance numbers, new tooling, report rewrites.

## 2. The code — one repository, `exabit-io/mx-llama.cpp`; substrate = its `master` (lead, 2026-09-24)

**All gfx906 code lives in [`exabit-io/mx-llama.cpp`](https://github.com/exabit-io/mx-llama.cpp)**, our fork of
mxxm-t's fork. **Its `master` is the substrate** for every build, branch and measurement. Today `master` =
`528384980`:

- **mxxm-t's fork** (`mxxm-t/mx-llama.cpp`, Marko Tombak) at its master `eefc4e732` (2026-09-21): 182 commits over
  upstream b10760, **kept as individual commits** (a merge, not a squash — any one can be named, reverted and binned);
- merged with **llama.cpp v0.5.0** (`7fe450e`, b11146) in `65027084e`: 13 conflict hunks and 3 silent breaks combined so
  no fork feature is lost (listed in that commit's message and in PR #17);
- plus `528384980`, `GGML_HIP_RCCL=ON` by default.

It is offered to mxxm-t as [mxxm-t/mx-llama.cpp#17](https://github.com/mxxm-t/mx-llama.cpp/pull/17), opened from
branch `merge-v0.5.0` (same commit as `master`; deleted once merged). When mxxm-t merges it, `master` tracks mxxm-t's
master. **Every upstream release, one procedure:** merge it into `master`, offer that to mxxm-t as a PR, move the four
`gfx906-*` branches onto the new `master`.

Verified before the PR (`night-20260919/pr-test.sh`): build 0 warnings; `test-backend-ops` 16273/16273 on two dies,
16272/16273 on two (upstream's `ADD_ADD` f16 test at its 1e-7 edge — 0/352 on re-run, 4/352 without the merge: an upstream
flake); perplexity 16K/6 **5.6153**; `-sm layer` generates; MTP `draft-mtp` 156/197 accepted; Flash-Next loads and
generates at production flags.

### Branches of `exabit-io/mx-llama.cpp` — six, nothing else

| branch | commit | contents |
|---|---|---|
| `master` | `528384980` | **the substrate** |
| `merge-v0.5.0` | `528384980` | PR #17's branch; deleted when merged |
| `gfx906-both` | `a23e12438` | master + AR size gate (term 06) + `GGML_TP_AR_MAX_NE` default 20481 + `GGML_CUDA_FA_QUANTS=all` default |
| `gfx906-single`, `gfx906-multi` | `a23e12438` | = gfx906-both until single-/multi-only winners are binned |
| `gfx906-candidates` | `d19af19e8` | master + our 20 code patches (+ docs, `BRANCHES.md`), not yet binned — the source of round-1 arms |

Each branch's own commits are its bin (`git log gfx906-both ^master` is the `both` set). Binned `both` and in
`gfx906-both`, not re-tested (lead, 2026-09-24): `tp-ar-size-gate` (own commit; default 20481 because mxxm-t
`41c46cedb` turned custom AR on by default and the fork's 262144 threshold is the configuration measured at -19% prefill),
`custom-allreduce` and `q8-repack` (in master), `rccl-collective` (master's own default).

**Two separate upstreams — never conflate them.** mxxm-t's fork is the kernel substrate. **mixa3607's `ML-gfx906`**
builds ROCm for gfx906 and publishes Docker presets; it has no llama.cpp source. Its one llama.cpp patch
(`mxxm-gfx906-kcase.patch`, Q4_K/Q5_K/Q6_K MMQ configs — mxxm-t PR #3, closed unmerged) and its compiler flag
`-mllvm -amdgpu-sched-strategy=max-ilp` are **not** in master; both are candidate patchsets to bin.

**Retired:** `exabit-io/llama.cpp` (our older fork of ggml-org) holds the history before 2026-09-24 — 27 branches and all
the `gfx906/v0.4.1/*`, `gfx906/v0.5.0/*`, `import/*` tags. **Archived 2026-09-24** (read-only, everything kept, README
and description point here). Unarchiving is possible if it is ever needed.

## 3. Fixed test conditions — identical for every patchset, never varied

| item | value |
|---|---|
| base | `gfx906-both` of `exabit-io/mx-llama.cpp` (section 2) |
| build | `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGGML_HIP_RCCL=ON -DGGML_CUDA_FA_QUANTS=all -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON`, Release, ccache — same for every arm, checked by `assert-arms-comparable.sh` |
| model (perf) | Qwen3.8-27B Q8_0, four dies, `-sm tensor -ngl all -fa on` |
| KV | **f16 / f16** |
| multi-user cell | **4 x 64K** (`-npl 4 -npp 65536 -ntg 1024`) |
| single-user cell | **1 x 255K** (`-npl 1 -npp 260864 -ntg 1024`, context 262144 = `n_ctx_train`) |
| power | 125 W per die, host RAPL 150 W, clamp watchdog + SMC log |
| collective | RCCL (`GGML_CUDA_ALLREDUCE=nccl`, topology file) + custom AR gated at 20481 |
| compat gate | Qwen3.8-Flash-Next, production flags (`--n-cpu-moe 41`, mlock, `LLAMA_PLE_SHARD=1`), greedy 64 tokens: pass/fail only |

Measured cell times at f16: 4x64K = 6.9 min, 1x255K = 11.9 min. One arm, both axes, n=5 = 94 min.

## 4. The loop

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

## 5. Round 0 — the substrate is verified

Done: the PR verification of section 2 is the substrate's correctness gate. Before round 1 measures, the same
gate runs once on the `gfx906-both` build (AR gate + FA_QUANTS defaults on top); `binrun.sh` refuses to
measure without it.

## 6. Round 1 — our patchsets + mixa3607's build flag (7 arms + base, ~11 GPU h)

Arms built on `gfx906-both`, commits cherry-picked from `gfx906-candidates` (term numbers as in `terms/`):

| patchset | terms | measured against | note |
|---|---|---|---|
| mmvq-q8-fastpath | 01 05 09 15 | base | one unit: 01 does not compile without 05 (`q8_fast`); 15 needs 01's define and is its MUL_MAT_ID fix |
| norm-add-fusion | 02 03 | base | |
| gdn-producer-fold | 04 07 08 10 12 | norm-add-fusion | cannot apply without 02 03; binned on its increment |
| s1b-repacked-matvec | 17 18 19 | base | |
| fa-head256-rows | 22 27 | base | an n=1 read on the earlier v0.5.0 base was -11% decode at 4x64K: watch, not a verdict |
| dpp-warp-reductions | 23 28 | base | |
| max-ilp | (build flag) | base | mixa3607/ML-gfx906's `-mllvm -amdgpu-sched-strategy=max-ilp`, base code |

Not patchsets for this instrument: terms 11 13 14 16 20 25 29 (docs, build script, inventory); 21 24 26 (MTP,
R3.9). Terms 24 (adaptive MTP, upstream PR #27210) and 26 were set aside because the old squashed substrate had
lost the fork's MTP code; `master` has it back, so both are to be ported to `gfx906-candidates`. Scripts: `night-20260919/binrun.sh`, `binstats.py`, `fncompat.sh`.

## 7. Round 2 — the substrate's patchsets (~8 arms + base, ~14 GPU h)

The substrate now keeps mxxm-t's 182 commits individually, so a feature can be removed by reverting its own
commits on top of `gfx906-both` (or switched off at runtime where the fork provides a switch). Classified from
commit subjects and switch code, 0 GPU; each group's commits are re-checked against the substrate before it is
built:

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

**C. Not ours to measure — recorded by inspection, 0 GPU.** Other models (DeepSeek-V4/V4.1, DSpark, DFlash,
MiniMax, MiMo2, gemma), MXFP4 (not a quant we run), MTP/speculative (R3.9, `technique-requires-implementation`),
pipeline-parallel-only scheduling (we run one stage), out-of-memory fallbacks (robustness, keep), BF16-in-F32
(`upstream-already-has-it`, PR 28846), docs and CI.

mixa3607's kcase patch and the 34 later mxxm-t commits' Flash-Next/K-quant items (`afca10209`, `2e740434e`,
`d9c6fc44d`) belong to group B (the 27B is Q8_0 and never runs K-quant MMQ); the 27B-visible later commits
(`de27e7509`, `20af9a480`, `62d4be47d`, `be8ff98aa`, `a355590d2`, `cf6a98f73`, `54702a718`, `27f755681`) join group A
after the per-group check.

How a substrate bin lands: the substrate stays whole in `master`. A patchset that regresses on a
profile is switched off in that profile's launch settings (runtime switch) or reverted on that profile's branch.

## 8. Timeline — starts on the lead's go (nothing is queued)

| step | GPU h |
|---|---:|
| gate on `gfx906-both` | ~0.3 |
| round 1 multi-user bins | 4.6 |
| round 1 single-user bins + compat | 8.2 |
| round 2 multi-user bins | 5.2 |
| round 2 single-user bins + compat | 9.2 |

If a phase overruns by more than 20%, stop and report. Every launch gets a PID waiter and a monitor in the same turn.

## 9. Versioning — nothing is thrown away

The branches move; every state is pinned by an **annotated tag**:

| where | tag | state |
|---|---|---|
| `exabit-io/mx-llama.cpp` | `gfx906/v0.5.0/r0/{master,both,candidates}` | **current**: round 0 on v0.5.0 |
| `exabit-io/mx-llama.cpp` | `gfx906/v0.5.0/r<N>/…` | after round N's bins are committed |
| `exabit-io/llama.cpp` (retired) | `gfx906/mx-merge-v0.5.0/*`, `gfx906/v0.5.0+mxxm-t-eefc4e732/*`, `gfx906/v0.5.0/*`, `gfx906/v0.4.1/*`, `import/*`, `gfx906-20260909`, `gfx906-b11067-1d1361e`, `gfx906-pre-v041-20260920` | history |

All measurement data, including runs later judged invalid or stopped, is mirrored to `data/raw/night-20260919/`.
GitHub releases mark milestones only.

## 10. What went wrong — do not repeat

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
7. **Conflating two upstreams.** "mxxm" (mxxm-t's fork) and mixa3607's ML-gfx906 were written about as
   one source for an hour, which nearly ended the project. Name both in full, always.
8. **Replacing the substrate three times in one night** (v0.4.1 squash -> v0.5.0 squash -> v0.5.0 + cherry-picks
   -> this merge). Each change was reasoned, but the churn itself made the state unreadable. The substrate now
   changes only by the lead's decision, recorded in section 2.

## 11. Where things are

| what | where |
|---|---|
| **code (the only repo)** | `github.com/exabit-io/mx-llama.cpp` — `master` (substrate), `gfx906-both/-single/-multi`, `gfx906-candidates`; local clone `/root/exabit-llama.cpp`, remote `exabit-mx` |
| PR to mxxm-t | https://github.com/mxxm-t/mx-llama.cpp/pull/17 |
| plans, docs, data, records | `github.com/exabit-io/llama.cpp-gfx906-tuning` (this repo) |
| retired code history | `github.com/exabit-io/llama.cpp` (archived 2026-09-24, read-only) |
| run scripts | `/root/night-20260919/binrun.sh`, `binstats.py`, `fncompat.sh`, `gate-v050.sh`, `pr-test.sh` (copies in `tools/binrun/`) |
| records | `survey/*.md`, lint `survey/survey-lint.py`; records carry the base they were measured on |
| requirements | `REQUIREMENTS.md` |

## 12. History of this plan (2026-09-24)

1. Rebased onto v0.5.0 with mxxm-t's 0c81bd502 squashed (v0.4.1 method) — superseded: the fork had moved on.
2. Added mxxm-t's 34 newer commits as cherry-picks + mixa3607's kcase patch — superseded by the lead's directive
   to use the upstreamable merge instead, which keeps every fork feature the squash had set aside (the fork's
   MTP `process_decode`, chunked MMQ, the server speculative reset).
3. Merged v0.5.0 into mxxm-t's master on `exabit-io/mx-llama.cpp`, opened PR #17, and made that the substrate.
4. Consolidated all code into `exabit-io/mx-llama.cpp` (substrate = `master`, six branches); `exabit-io/llama.cpp`
   retired as history (lead: "reduce and simplify things greatly").
