Merges the llama.cpp v0.7.0 release (`620d222`, b11669) into `master`. It is one merge commit on top of your `master` at `c4d12e2c0`, the same way you merged b11553. The branch fast-forwards from `master`, and the PR contains no other changes.

This replaces #17. Your b11553 merge made that PR obsolete. Unlike #17, this one does not change the RCCL default; `GGML_HIP_RCCL` stays `OFF` as in your tree.

## What v0.7.0 brings

There are 116 upstream commits since b11553. The ones that touch code this fork changes:

- #29353: an MMA-based chunked GDN kernel (NVIDIA Ampere GA10x, Ada and Blackwell; on HIP, RDNA3.5 only).
- #30506: LiliCorr speculative drafting.
- #30405: model buffers map only the file ranges of their own context. `llama_buf_map` is gone.
- #30329: the scheduler is created in `llama_context::sched_new()` and reports graph reallocations.
- #30453: meta-backend subgraph recreation, which matches your existing fix.
- #30014: hidden default symbol visibility.

## How the merge was resolved

Every fork feature is kept. Where both sides changed the same code, the two changes are combined, not chosen between.

Textual conflicts (10 hunks in 8 files):

- `common/speculative.cpp`: upstream's LiliCorr setup is placed ahead of the fork's fallback to the target vocab's mask token.
- `ggml/src/ggml-backend-meta.cpp`: #30453 is the fork's own fix, so the fork's comment is kept.
- `ggml/src/ggml-cuda/common.cuh`: the fork's context fields are kept, with upstream's `= {}` initializers for the stream and cuBLAS arrays.
- `ggml/src/ggml-cuda/gated_delta_net.cu`: both the fork's chunk kernel header and upstream's MMA header are included.
- `src/llama-context.cpp`: both call sites use upstream's `sched_new()`, and the fork's sync-non-graph-inputs policy moves into it.
- `src/llama-ext.h`: the fork's MTP prefill and pre-norm accumulation declarations are kept.
- `src/llama-model.cpp`: the fork's skip for contexts with nothing to allocate is kept, without the removed per-file buffer map. `load_tensors()` ends with `load_tensors_hadamard()` and then the PLE prefault.
- `tools/server/server-context.cpp`: `release()` resets the speculative state as before, and also drops the prompt of stateless tasks as upstream now does.

Two places merged without a textual conflict but would have been wrong, so the merge commit changes them:

- **LiliCorr with the deferred DFlash prompt encode.** LiliCorr stores the projected rows of each inject in the eager path only. The fork's deferred encode replays injects in `flush_pending_upto()` without that step, so a LiliCorr drafter would draft from empty state. A LiliCorr drafter now stays on the eager path.
- **The new MMA GDN kernel with the snapshot ring.** The MMA kernel writes only the final state, while the recurrent-snapshot ring needs states 1..K-1 too. When `K > 1` the call goes to the fork's kernels, as before. On gfx906 the MMA path is never taken.

## Testing

All tests ran on a Mac Pro 7,1 with 2x Radeon Pro Vega II Duo (4x gfx906, 32 GB each), ROCm 10.0.0-gfx906+20261018172427 (HIP 7.15.26333) and kernel 7.0. The build used the tree's own defaults (`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906`, RCCL off), with GPUs capped at 125 W.

- **Build:** 0 errors. All 7 warnings come from fork code this merge does not change: an unused variable in `delta-net-base.cpp`, an unused parameter in `deepseek41.cpp`, and five occupancy notes from the q8_repack GEMM.
- **`test-backend-ops`:** every op passes on every die, 18644/18644 on each of the four.
- **Perplexity** (Qwen3.8-27B Q8_0, wikitext-2, 16K x 6 chunks, `-sm tensor`): **5.6122 +/- 0.0623**. #17 measured 5.6153 and the pre-v0.5.0 builds 5.6171.
- **`-sm layer`** with Qwen3.8-27B: greedy generation is coherent.
- **MTP** (Qwen3.8-27B, `llama-server --spec-type draft-mtp --spec-draft-n-max 2`, `-sm tensor`): 256 tokens generated, **153 of 203 drafts accepted (75%)**, 69.2 tok/s.
- **Qwen3.8-Flash-Next** (qwen4exp) UD-Q4_K_XL (`-sm tensor --n-cpu-moe 41 -lm mlock`, `LLAMA_PLE_SHARD=1`): loads, and greedy generation is coherent.
- **`-lm dio`**, about @xmesaj2's comment on #17: on this branch the model file is opened with `O_DIRECT` (checked with strace) and no read falls back to buffered I/O, on both the GPU and the CPU load path. I could not reproduce the fallback on any build, including #17's branch, a pre-v0.5.0 build, and your master `c4d12e2c0` (built with our gfx906 patches on top, which do not touch the loader). The fallback warning names the errno (`EINVAL`/`EFAULT` from `read()`, or the `open()` error), and that would show what differs on the affected setup.

**Not tested** (no hardware or models here): CUDA and NVIDIA builds (which includes the new MMA GDN path), AMD architectures other than gfx906, DeepSeek-V4/V4.1, DSpark, DFlash, LiliCorr, gemma, MiniMax, and `-tps` pipeline stages.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
