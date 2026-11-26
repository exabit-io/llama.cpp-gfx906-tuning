Merges the llama.cpp v1.1.0 release (`1715275`, b12774) into `master` at `557a6068f`, plus three small fixes, two that the tests on gfx906 asked for and one found while comparing the two merges. The branch is the merge commit, then the fix commits; it fast-forwards from `master` and contains nothing else.

This replaces #24. You merged b12500 yourself in the meantime, so #24 is obsolete and this branch starts from your tree: where your merge and ours had resolved something differently, yours is what this branch has. `GGML_HIP_RCCL` stays `OFF`.

## What v1.1.0 brings on top of b12500

274 upstream commits. The ones that touch code this fork changes:

- **#31669, "GDN: use replay instead of rollback".** For the delta-net models (qwen35, qwen35moe, qwen4exp, clover-moe and the KDA models) upstream removed the snapshot branch: with `n_rs_seq > 0` a context keeps one state plus the inputs of the last ubatch, and the gated delta net op replays the kept tokens before the new ones (`ggml_gated_delta_net_ext`, `cparams.rs_replay`). This is the one decision of the merge, see below.
- #32901: the flash-attention kernels share one data-loading path and the instances are generated per K/V type (`fattn-tile-instance-f16-f16-dkq256-dv256.cu`, ...).
- #30685, #31898: host inputs of a split can be copied with `set_tensor_async` (a device capability, CUDA only), and `graph_optimize` can reserve scratch memory.
- #32454, #32500: fused NORM + MUL (+ ADD) and RMS_NORM + MUL + MROPE.
- #32065, #32505, #32805: few-row MMQ for `MUL_MAT_ID`, duplicate expert ids, and MMQ instead of MMVQ for Q4_K at 8 columns on VEGA20.
- #32650: the unfused lightning indexer graph is gone.
- The server's web UI is no longer in the tree (see "Building" below).

## How the merge was resolved

19 files had textual conflicts; the merge commit message lists each one.

**Rollback of the recurrent state.** Your snapshot ring and upstream's replay do the same job, so they are kept side by side, one per context:

- a context the ring covers (one sequence, a verified backend) keeps the ring and its snapshots, unchanged;
- every other context of these models with `n_rs_seq > 0` keeps the plane-indexed snapshots it has today;
- `LLAMA_RS_REPLAY=1` gives those contexts upstream's replay.

The default is the fork's own rollback because your gated delta net kernels do not replay, so a replay op runs on upstream's recurrent kernel. In one session here at `-np 4` with MTP (`--spec-draft-n-max 3`, Qwen3.8-27B Q8_0, `-sm tensor`) replay ran the prefill of a 2,089-token prompt about 15 % slower than the snapshots and decode 2 to 3 % slower, and an 11-token prompt at less than half the rate. (The GPU clocks of this box were held at 1000 MHz during that session, so these are ratios, not rates; I will repeat it at full clock and post the rates here.) What replay buys is memory, and that does not depend on the clock: 653 MiB of recurrent state instead of 2,394 MiB at `-np 4` with three rollback slots. Both give the same text when the snapshot arm is put on the recurrent kernel too (`GGML_GDN_NO_COLTHREAD=1 LLAMA_GDN_NO_CHUNK_SNAPSHOTS=1`), and upstream's `replay` case of `test-llama-context` passes with `LLAMA_RS_REPLAY=1`, so flipping the default is one line in `llama_context` if you prefer the memory.

`llama_memory_recurrent::ring_supported()` is the ring's condition as one static function, used by the memory and by `llama_context`, so the two cannot disagree. An op with replay inputs runs on upstream's recurrent kernel: your column-thread and chunked kernels do not replay pending tokens and do not store the inputs of the new ones.

**Scheduler.** Your staged input copies stay as they are. Upstream's `ggml_backend_sched_copy_input()` and its input prefetch are not used (you had left the prefetch out at b12500), and the drain before a reused graph (`LLAMA_REUSE_NO_DRAIN`) stays, because upstream dropped it for the input copy this branch does not take. Upstream's scratch tensors are in.

**The rest:** the `clamped` template switch of your norm kernels and upstream's new fusion switches are separate template parameters; the fixed-order top-k gather stays; `fattn-vec.cuh` takes upstream's mask strides, which carry what your `s31` computed; your DeepSeek-V4.1 graph stays (upstream's one change to the shared loader merged by itself); `LLAMA_NO_FUSED_LID` goes with upstream's unfused graph.

These merged without a textual conflict but needed a change:

- `cparams.rs_replay` is decided after your clamp of `n_rs_seq` for a small `n_ubatch`. Upstream sets it earlier; with the clamp it would have stayed true at `n_rs_seq` 0.
- Upstream's new `build_attn_iswa_impl()` is a free function; your replica-aware cache helpers (`build_cpy_k`, `build_get_k`, ...) are called through `gctx` there.
- The replay's conv-window input is an input of its own instead of a view, and `set_input_rep()` writes through `ggml_backend_tensor_set`, as your `s_copy_main` / `s_copy_tail` do.
- `ggml_gated_delta_net(..., K)` is not deprecated here: the ring builds it with K > 1.
- Upstream removed the K > 1 gated delta net cases from `test-backend-ops` with its snapshot branch. Six of them are back as `test_gated_delta_net_snapshots`, since the ring still uses that form.

## The fix commits

**Fused top-k MoE softmax on 64-lane rows.** `softmax_warp_inplace()` reduces each group with `warp_reduce_max()` / `warp_reduce_sum()` at their default width of 32. Since #31617 a row of that kernel is one physical warp, 64 lanes on GFX9, and a width-32 shuffle stays inside one half of the wavefront, so the maximum and the denominator leave out every expert whose index mod 64 is 32 or more. The selected ids do not change; without weight normalisation the selected weights are too large, with it the factor cancels. On gfx906 your `master` fails 68 `TOPK_MOE` cases of `test-backend-ops` on every die (the softmax routers with 128, 256, 288, 384 and 512 experts), and so does the plain merge. Both reductions now name the row width, as the `with_norm` block of the same kernel already does: 692/692. Nothing changes where a row is 32 lanes.

**576/512 attention without a tile instance.** For value heads above 256 the tile kernel only exists in the forms that fold several query heads into a block (2, 4 or 8 for 512/512; 4 or 16 for 576/512), but the kernel selection only asked for a GQA ratio of at least 2. v1.1.0 adds the first test model that has 576/512 with a ratio of 2 (`hy_v4_flash`), and `test-llama-context` aborts on it at `fattn-tile.cuh:1230` on a device without tensor cores; stock v1.1.0 does the same here. The selection now returns `BEST_FATTN_KERNEL_NONE` for those shapes, so the op goes to the next backend.

**Carried MTP row at position 0.** The first token of a sequence is paired with the hidden row carried over from the previous batch, and a sequence that starts at position 0 gets zeros (017a5d3f8). With device rows the carried rows are uploaded before the token loop and the zeroing sits inside it, so a new request is paired with the last row of the slot's previous request. The text does not change (the target verifies every draft), the draft does: the same greedy request gives 150 of 207 drafted tokens accepted on a fresh slot and 151 of 205 every time after it. With the reset ahead of the upload it is 150 of 207 every time.

Each fix is its own commit, so you can take or drop them one by one.

## Not changed, for you to look at

These are in `master` today and are not touched by this branch. They are read from the code, not measured, unless said.

- **Drafters on the deferred path.** The deferred replay (`flush_pending_upto`) never calls `store_context_features`, which LiliCorr needs and, since v1.1.0, upstream's DSpark Markov head as well. The server's default hides it: with prompt checkpoints on, the last four prompt tokens are decoded as their own small batch, which goes through the eager path. With `--ctx-checkpoints 0` the first draft after a prompt would read features that were never stored (LiliCorr) or produce no draft (DSpark Markov). Read, not run.
- **KV-only MTP store.** It mirrors the store block of `build_attn` but does not divide K and V by the scale of an FP8 cache. Needs an FP8 draft KV cache and a GGUF with scale tensors; the Qwen3.8 file here has none. Read, not run.
- **Snapshot ring and the kernels that write one state.** The op's contract lets a backend leave slots 1..K-1 unwritten for a ubatch longer than K, and the ring copies K slots from every ubatch. Two kernels write slot 0 only: upstream's MMA kernel (launched without a K check; taken on NVIDIA Ampere and later, CDNA and RDNA3.5, not on gfx906) and the chunked kernel, whose `keep_rs_t` parameter is unused (taken when the column-thread kernel declines: a head size other than 128, or a device without 64-lane warps). A rollback into such a ubatch would restore unwritten slots. Not reached by the 128-head models on gfx906 that were run here. Read, not run.
- **Repacked `MUL_MAT_ID` and duplicate expert ids.** `q8_repack/mmid-bounds.cuh` keeps the ids helper from before #32505, which made duplicate ids legal. No graph in the tree produces them today.
- **DeepSeek-V4.1.** The branch keeps your implementation and its GGUF layout. Files in upstream's layout (the output of `convert_hf_to_gguf.py` in this tree, and the published quantizations) do not load: the loader asks for the fork's tensor names and ends in "wrong number of tensors". Three upstream commits of this range are V4.1 work on upstream's graph and have nothing to land on here. If you want both layouts to load, say so and that can be a PR of its own.
- **KV replicas and sequence state copy** (unchanged since v0.7.0, see Testing): `gemma4-dense` and `mistral-large4-moe` fail `cp_h` / `cp_d` of `test-llama-context` when their KV-sharing layers sit on different dies.

## Building

Two things stopped a build here before any code was involved; neither is fixed in the tree by this PR:

- The server's web UI is downloaded at build time from upstream's bucket by build number (`b<commit count>`). The fork's count names no upstream build (b12717 on `master`: HTTP 404), the script falls back to `latest`, and today's `latest` no longer passes the asset check of a b12500 tree (`UI: missing required asset(s)`). `HF_UI_VERSION=b12500` for `master`, `HF_UI_VERSION=b12774` for this branch.
- With a `gcc-14` installed without `libstdc++-14-dev`, the HIP compiler selects that directory and fails at `'cmath' file not found`; `-DCMAKE_HIP_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13` pins it.

## Testing

All tests ran on a Mac Pro 7,1 with 2x Radeon Pro Vega II Duo (4x gfx906, 32 GB each), ROCm 10.0.0-gfx906+20261018172427 (HIP 7.15.26333, LLVM 23) and kernel 7.0, GPUs capped at 125 W. The builds used the tree's own defaults (`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906`, RCCL off).

Speeds are left out of the table: the GPU clocks of this box were held at 1000 MHz for most of the session (a power-envelope latch of this machine, unrelated to the code), which changes no result below but makes every rate meaningless. I will post rates after a restart.

| Check | This branch | `master` (`557a6068f`) |
|---|---|---|
| Build (`HF_UI_VERSION` set, see Building) | 0 errors; warnings: the q8_repack occupancy notes and one unused parameter in `deepseek41.cpp` | 0 errors |
| `test-backend-ops`, every op | 21871/21871 on each of the four dies | 20565/20633 on each: 68 `TOPK_MOE` cases |
| Same, the merge without the fix commits | 21803/21871: the same 68 | - |
| Perplexity, Qwen3.8-27B Q8_0, wikitext-2, 16K x 6 chunks, `-sm tensor` / `-sm layer` | 5.6130 / 5.6112 | 5.6126 / 5.6120 |
| Same, stock llama.cpp v1.1.0 / b12500 | 5.6070 / 5.6071 (v1.1.0) | 5.6186 / 5.6102 (b12500) |
| `-sm layer -lm dio`, greedy 64 tokens | the same text as `master`, 0 buffered fallbacks | - |
| MTP, `llama-server --spec-type draft-mtp --spec-draft-n-max 2`, 256 tokens | 150/207 drafted tokens accepted, on a fresh slot and on every later request | 154/200 |
| MTP row paths (default, host rows, device rows from the start, eager) | identical greedy text and 150/207 on all four | identical text and 154/200 on all four |
| Rollback, MTP `--spec-draft-n-max 3`: ring (`-np 1`), ring with `LLAMA_RS_REPLAY=1`, snapshots (`-np 4`), replay (`-np 4`, `LLAMA_RS_REPLAY=1`), snapshots on the recurrent kernel | the two ring arms identical; replay and snapshots-on-the-recurrent-kernel identical on all four requests; no refused rollback; the log names the mode of each arm (`rs_replay = true` only in the replay arm, 653 MiB against 2,394 MiB) | - |
| `test-llama-context`, 150 test models, four dies, default rollback | 148 pass; `gemma4-dense` fails `cp_h`, `cp_d`, `frag` and `mistral-large4-moe` fails `cp_h`, `cp_d` (KV replicas, see above); `rollback` 16/16, `rollback_bound` 16/16, `replay` 15/15, `shared` 25/25 | 144 of 145 (`gemma4-dense`) |
| Same with `LLAMA_RS_REPLAY=1` | the same 148 and the same counts | - |
| Same, stock v1.1.0 | aborts on `hy_v4_flash` (model 61 of 150) | - |
| Qwen3.8-Flash-Next, `-sm tensor`, greedy 64 tokens | coherent | coherent |

The perplexity moved a little, and so did the greedy text of the MTP test after about 80 tokens. Stock llama.cpp moved by more between the same two points (third row of perplexities), so I read this as upstream's numerics and not as something the merge did.

DeepSeek-V4.1 and DSpark were not run: the file here is in upstream's layout (see "Not changed").

🤖 Generated with [Claude Code](https://claude.com/claude-code)
