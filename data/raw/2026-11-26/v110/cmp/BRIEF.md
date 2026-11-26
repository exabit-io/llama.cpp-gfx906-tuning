# Brief: how mxxm-t merged upstream to b12500, compared with how we did (PR #24)

Read-only analysis. Do not build, do not use the GPUs, do not fetch or push, do not check out or modify any
worktree. Write only your one findings file under `/root/rebase-v110/cmp/`.

## The repositories and commits

Git repository `/root/exabit-llama.cpp` (use `git -C /root/exabit-llama.cpp ...`; every commit below is in it).

- Upstream = ggml-org/llama.cpp. `b1b770d` = v1.0.0 = build b12359. `aaa807c56` = build b12500. `v1.1.0` = b12774.
- mxxm-t/mx-llama.cpp = a fork for AMD gfx906 (Vega 20, HIP, 64-wide wavefronts, four dies) with its own features:
  a tensor-parallel "meta" backend (`-sm tensor`), a snapshot ring for recurrent state (K slots, `keep_rs`), deferred
  prefill for the MTP / DFlash drafters, the Q8_0 "repack" kernels (`ggml/src/ggml-cuda/q8_repack/`), KV replicas,
  DeepSeek-V4 / V4.1 / DSpark support, and more (`FEATURES.md` in his tree).
- OUR merge (pull request #24 to him, never merged): branch tip `a73c601e6` =
  merge commit `f6ec0b7c3` (upstream v1.0.0 onto his then-master `cdc81dae4`)
  + `7334d8c27` (HIP: expert-group ballot on 64-wide wavefronts) + `a73c601e6` (HIP: PAD_REFLECT_1D in row chunks).
  It was stacked on our earlier merges (v0.7.0, v0.9.0), whose resolutions it carries.
- HIS merges, done independently of ours: `6192a1c88` (Merge tag 'b12169') followed by his fix commits `0c3567111`,
  `977ff5061`, `ccc0c0d2d`, `1c8f7fa09`, `773721de3`; then `897e26808` (Merge tag 'b12500', second parent `aaa807c56`)
  followed by `9284233fe`, `5ab83d630`, `04dbc7b25`, `557a6068f` (= his master today).

## What is already set up

`/root/wt-cmp-ours-b12500` is a worktree holding OUR `a73c601e6` with upstream b12500 merged in, NOT committed.
For every file that merged cleanly there, it is "our handling carried to b12500", so

    git -C /root/wt-cmp-ours-b12500 diff 897e26808 -- <file>

shows his merge (`-` lines) against ours (`+` lines) with upstream's own changes cancelled out.

Eight files have conflict markers in that worktree (upstream changed the same lines between b12359 and b12500):
`common/speculative.cpp`, `ggml/src/ggml-cuda/fattn-vec.cuh`, `ggml/src/ggml-cuda/norm.cu`,
`ggml/src/ggml-cuda/pad_reflect_1d.cu`, `ggml/src/ggml-cuda/top-k.cu`, `ggml/src/ggml-cuda/topk-moe.cu`,
`src/llama-model.cpp`, `tools/server/server-context.cpp`. For those compare the fork deltas instead:
`git diff aaa807c56 897e26808 -- <file>` (his delta at b12500) against `git diff b1b770d a73c601e6 -- <file>`
(ours at b12359), with `git diff b1b770d aaa807c56 -- <file>` for what upstream did in between.

His commits after the merge are NOT in that diff: read them with `git show <commit>`.

## Where our decisions are written down

- `git -C /root/exabit-llama.cpp log -1 --format=%B f6ec0b7c3` (v1.0.0 merge message: every conflict and how it was combined)
- `git -C /root/exabit-llama.cpp log -1 --format=%B 13e8f8b3e` (v0.9.0 merge message)
- `/root/llama.cpp-benchmarking/RELEASE-RUNBOOK.md` section 3: three tables of "silent breaks" (things that merged
  cleanly and were wrong), one per cycle
- `/root/llama.cpp-benchmarking/reports/2026-11-14-v100-rebase.md`, `2026-11-06-v090-rebase.md`, `2026-10-22-v070-rebase.md`

## What I need, per file or per topic

1. What he did. 2. What we did. 3. One verdict:
   - `SAME`: only wording, order or naming differs.
   - `EQUIVALENT`: different code, same behaviour (say why you believe that).
   - `HIS-BETTER`: he fixed or covered something ours missed, or his is more robust. Say what.
   - `OURS-FIXES-HIS`: his tree has a defect ours does not. Give the concrete failure scenario (which model or
     graph shape, which flags, which hardware) and say whether it is reachable on gfx906 (Vega 20 is neither CDNA
     nor RDNA3.5; HIP; wave size 64; this box serves Qwen3.8-27B Q8_0 with an MTP head under `-sm tensor`, and tests
     Flash-Next, DeepSeek-V4.1 with DSpark, Nemotron with LiLiCorr).
   - `UPSTREAM-MOVED`: the difference exists only because upstream rewrote the code between b12359 and b12500.
   - `UNSURE`: say what you would need to know.
4. How it could be tested on the gfx906 box (a test binary, an environment switch, a model and flags), if at all.
5. Anything in his version that the 20 Exabit commits will collide with when they are carried onto his tree. Their
   subjects: `git -C /root/exabit-llama.cpp log --reverse --format='%h %s' a73c601e6..224b73e24`. They touch the fused
   RMS_NORM/MUL/ADD path and its Q8_1 emission (`norm.cu`, `ggml-cuda.cu`), the gated-delta-net producer folds
   (`gated_delta_net.cu`), DPP warp reductions on HIP (`common.cuh`), MMVQ / MUL_MAT_ID widths (`mmvq.cu`,
   `ggml-cuda.cu`), the tensor-parallel all-reduce size gate, and `ggml/src/ggml-hip/CMakeLists.txt`.

Rules for the findings: read the code before you claim anything, and give `file:line` (in his tree at `557a6068f`
unless you say otherwise) for every claim. Do not guess; an honest `UNSURE` is worth more than a confident wrong
verdict, because these verdicts decide what goes into a pull request to him. Our handling is not presumed right.

Write the findings to the file named in your task (Markdown, one section per file or topic, most important first),
and return a summary of at most 400 words: the `OURS-FIXES-HIS` and `HIS-BETTER` items with their scenarios, then
counts of the rest.
