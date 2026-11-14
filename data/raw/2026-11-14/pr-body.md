Merges the llama.cpp v1.0.0 release (`b1b770d`, b12359) into `master`, plus two HIP fixes for upstream bugs that show on gfx906. The branch is the merge commit on `cdc81dae4`, then the two fixes. It fast-forwards from `master` and contains no other changes.

This replaces #23, which merged v0.9.0 and was not merged. v1.0.0 contains v0.9.0, and this merge keeps #23's resolutions (the tree was built on top of them), including its one exception: DeepSeek-V4.1 uses upstream's implementation with your tensor-parallel routing. `GGML_HIP_RCCL` stays `OFF` as in your tree.

## What v1.0.0 brings

There are 806 upstream commits since b11553, 249 of them since v0.9.0. The ones that touch code this fork changes:

- #31927: the MTP draft context reads the target's hidden rows from a device buffer instead of host copies.
- #31047, #32067: the MoE layout of MMQ/MMF is linearized. Tile specs replace the per-expert bounds, and the tile width is picked against the tokens per expert on every architecture. #31533 adds few-row MMQ, which is NVIDIA only (it is compiled out on HIP).
- #31084: the radix top-k picks a deterministic set and tie order.
- #31660, #32078: recurrent conv snapshots are stored through one helper (`build_conv_state_store`), the non-fused gated delta net is removed, and the graph node budget no longer depends on the ubatch.
- #31419, #30966, #32100, #32009: the scheduler skips inputs that nothing reads, prefetches the inputs of a following CPU split, and is reused across reserves.
- #32001, #31284, #32004, #30542: the CUDA graph structs move to `graph.cuh`, graph executables are re-instantiated when cluster dims change (CUDA only), a reused graph skips the compatibility check, and the VMM pool reservation can shrink.
- #31683, #31733: KV shift fix with per-model `build_rope`, and the model becomes a member of the graph context.
- #28392, #28413: FP8 (E4M3) KV cache and weights.
- #30565, #29801: deprecated API removed (`llama_set_warmup`, `--defrag-thold`, ...), and the NextN / layer-input / embedding-row API moves from `llama-ext.h` to `llama.h`.
- #31185: external PLE files (`--ple` is an alias of `--shard`), with a reworked split loader.
- #32102: the Qwen3.5 converter merges `attn_qkv`+`attn_gate` and `ssm_beta`+`ssm_alpha`; the loader and the `-sm tensor` split rules accept both layouts.

## How the merge was resolved

Every fork feature is kept and combined with upstream's change. Stacked on #23's tree there are 15 conflicted files; the merge commit message lists each one, followed by #23's list as it stands now. The main combinations:

- **MTP hidden rows:** your deferred prefill fills its accum buffer from the host rows, which upstream's device mode skips. The device rows therefore start when the deferred prefill has replayed and released its buffer, and from the start when it is not armed (more than one sequence, or `LLAMA_MTP_DEFERRED_PREFILL=0`). Your adaptive disable turns the device rows off together with the extraction and back on in `begin()`. `LLAMA_MTP_DEV=0` keeps the host rows. All four combinations give the same greedy output (see Testing).
- **MMQ:** your column chunking (dense) and token chunking (MoE) under workspace pressure compute the tile count and build the tile specs per chunk. `ggml_cuda_launch_mm_ids_helper` keeps an optional per-expert bounds output, because your repacked MoE kernels read the bounds.
- **Radix top-k:** upstream's fix still gathers the elements above the threshold with an atomic counter, so their order depends on scheduling. Your count/offsets/gather kernels (1b7d76ffc) write every index at a fixed position, so they are kept, on top of upstream's contiguous histogram.
- **Scheduler:** upstream's `hv_tensor_requires_copy` is the rule you already have as `inputs_read` (4d03828a1), which also follows a view to its root copy. Your rule is kept and upstream's bookkeeping dropped. Upstream's input prefetch is added to your inlined copy loop.
- **CUDA graphs:** your fallible instantiate and HIP kernel-set check use upstream's instance struct. Upstream's cluster-dim check (CUDA only) goes through the fallible path.
- **Conv snapshots:** your ring writes are kept in `delta-net-base.cpp` and `qwen4exp.cpp`. Upstream's `build_conv_state_store` serves the cases without the ring.

These merged without a textual conflict but would have been wrong, so the merge commit changes them:

- In device-row mode the host extraction that fills your deferred MTP accum buffer is skipped, so the replay would have injected zeros. `llama_set_embeddings_nextn_dev` now refuses while the accum buffer is set, and the accum buffer is refused while the device rows are on.
- Your KV-only MTP store mirrors the store block of `build_attn`, which now divides K and V by the scale of an FP8 cache. The KV-only store does the same.
- Your KV-only MTP replay still called `ggml_rope_multi` with the `sections` array that #31683 removed. It uses `build_rope` like the full path.
- `README.md` named b11553 as the base; it now names v1.0.0.

## The two fix commits

**Grouped top-k fusion on 64-wide wavefronts** (unchanged from #23). HIP's `__ballot_sync` ANDs the wavefront's ballot with its mask, so the second 32-lane row of a wavefront read the first row's group bits. Stock v1.0.0 still fails 91 of 692 `TOPK_MOE` cases on gfx906; with the fix 692/692 pass.

**`PAD_REFLECT_1D` with 2^24 or more rows.** The kernel runs one block of 256 threads per row along `grid.x`, and HIP rejects a launch with 2^32 or more threads along one grid dimension. #31877 added a test case with 16,777,217 rows, so stock v1.0.0 aborts there on gfx906 with "invalid configuration argument", which also ends a full `test-backend-ops` run at that point. On HIP the rows are now launched in chunks; CUDA is unchanged. A minimal HIP kernel on this box is accepted with 2^32 - 1 threads and rejected with 2^32.

Both are separate commits so you can take or drop them on their own.

## Points to check

- **MTP row path.** The default is your deferred prefill for the first prompt and device rows afterwards. On this box the four paths differ only in prefill: with your deferred prefill a 2,089-token prompt runs at 549-557 tok/s, without it at 534-539, and decode is 70.5-71.7 tok/s on all four (one server per arm). If you would rather keep the host rows whenever the deferred prefill is compiled in, `dev_rows_enable()` in `common/speculative.cpp` is the one place to change.
- **Top-k ties.** Your gather keeps the ties in block/thread order; upstream keeps the lowest columns. Both are deterministic, and `test-backend-ops` accepts either (it compares each backend with itself).
- **Graph node budget.** `graph_max_nodes()` no longer scales with the ubatch (#32078). The 16K perplexity run below uses `-ub 2048` and fits.
- **Merged Qwen3.5 tensors (#32102).** No GGUF from the new converter was available here, so the merged `attn_qkv_gate` / `ssm_beta_alpha` layout is untested on this branch. Files converted before v1.0.0 load as before.
- **FP8 KV cache.** Upstream refuses it for the architectures that read the cache directly. Upstream's `test-fp8` passes on this branch; an FP8 cache together with your KV replicas or the KV-only MTP store was not run.
- **KV replicas and sequence state copy.** See the `test-llama-context` note under Testing: with replicas active, a sequence copied through a saved state goes wrong at the second generated token on the gemma4 and mistral-large4 test models. It predates this PR.

## Testing

All tests ran on a Mac Pro 7,1 with 2x Radeon Pro Vega II Duo (4x gfx906, 32 GB each), ROCm 10.0.0-gfx906+20261018172427 (HIP 7.15.26333) and kernel 7.0, GPUs capped at 125 W. The build used the tree's own defaults (`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906`, RCCL off).

| Check | This branch | #23 (v0.9.0 merge) |
|---|---|---|
| Build | 0 errors; 5 warnings, all q8_repack occupancy notes in fork code this merge does not change | 0 errors, 6 warnings |
| `test-backend-ops`, every op | 20388/20388 on each of the four dies | 19351/19351 on each |
| Same, stock v1.0.0 | aborts at `PAD_REFLECT_1D` after 13418 cases | - |
| Perplexity, Qwen3.8-27B Q8_0, wikitext-2, 16K x 6 chunks, `-sm tensor` | 5.6126 +/- 0.0622 | 5.6126 +/- 0.0622 |
| `-sm layer -lm dio`, greedy 64 tokens | coherent, 0 buffered fallbacks | coherent, 0 fallbacks |
| MTP, `llama-server --spec-type draft-mtp --spec-draft-n-max 2`, 256 tokens | 154/200 drafts accepted, 68.3 tok/s on the first request after load, 70.7-71.7 on the next three | 154/200, 67.0 tok/s |
| MTP row paths (default, `LLAMA_MTP_DEV=0`, `LLAMA_MTP_DEFERRED_PREFILL=0`, both) | identical greedy output and 154/200 on all four; decode 70.5-71.7 tok/s on every path. With `-v` the two device modes allocate the 40 MiB row buffer on the `Meta(ROCm0..3)` device, the host modes none, and there is no fallback | - |
| MTP with `n_probs: 5` | every token, drafted or not, carries 5 `top_logprobs` (128 of 128) | every token carries 5 `top_logprobs` |
| MTP with `-np 4`, four concurrent requests | all coherent, 24.1-26.8 tok/s per request, 63-76% acceptance, no refused rollbacks (device rows on, four sequences) | all coherent, 24.4-26.9 tok/s per request |
| Qwen3.8-Flash-Next (qwen4exp) UD-Q4_K_XL, `-sm tensor --n-cpu-moe 41 -lm mlock`, `LLAMA_PLE_SHARD=1` | loads, greedy output coherent | coherent |
| `test-llama-context`, dummy models of every architecture, one worker | one die: every case passes on 144 models. Four dies: 142 of 144, see below | passes on 141 models |
| `llama-gguf-split`: split, then merged | greedy output of the merged file identical to the original | identical |
| `test-llama-archs` (builds and runs a dummy model of every architecture) | passes, 144 models | - |
| `test-fp8` on the FP8 dummy models: requantize, KV cache | both pass | - |

The build row is a clean build of the branch tip. The other rows were measured on a build of the same tree; the MTP rows are single servers (n=1), with the 256-token request repeated three times where a range is given. These are correctness and smoke values at small depths, not service measurements.

**`test-llama-context` on four dies.** `gemma4-dense` and `mistral-large4-moe` fail the sequence-copy cases (`cp_h`, `cp_d`), and gemma4 also the fragmented restore (`frag`). Both models share KV rows between layers, and on four dies the sharing layers land on different devices, so your KV replicas (de27e7509) are in use. With `LLAMA_KV_REPLICA=0`, or with one die visible, both pass, and stock v1.0.0 passes on four dies. The first generated token after the copy is right and the second is not. Builds of the v0.9.0 merge of #23 and of the v0.7.0 merge of #22 fail the same copy cases on four dies with the gemma4 model, so this predates these merges; your `master` has the replicas but not this test, which came with upstream later. #23 reported all 141 models passing; that run cannot have had replicas active. This PR does not change it.

As in #23, `test-llama-context` as ctest runs it (`GGML_SCHED_DEBUG_REALLOC=1`, four workers) stops on graph reallocations that come from the fork's scheduler; the rows above are direct runs with one worker.

DeepSeek-V4.1 (with DSpark) and Nemotron 3.5 Bolt (with its LiLiCorr drafter) are running on this box now, on stock v1.0.0 and on this branch; I will post the results here.

**Not tested** (no hardware or models here): CUDA and NVIDIA builds (including few-row MMQ, the fused MMQ paths, the cluster-dim check and the multi-node NCCL path), AMD architectures other than gfx906, DeepSeek-V4, DFlash, gemma, MiniMax, FP8 caches and weights on real models, GGUFs from the v1.0.0 Qwen3.5 converter, and `-tps` pipeline stages.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
