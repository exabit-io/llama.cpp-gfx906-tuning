# A2 - second read of the v1.1.0 merge and of master-v110 (ggml core, CUDA/HIP backend)

Trees: the merge `/root/wt-merge-v110` at `aa670811a`, master `/root/wt-master-v110` at `b6ee9c341`. Reading only: nothing
built, nothing run on a GPU. `file:line` names the tree. "Old" is our v1.0.0 master (`a73c601e6..224b73e24`).

How the ports were checked: for every file the 22 commits touch I compared the set of added and removed lines of the
old series with the new one (`git diff a73c601e6 224b73e24` against `git diff merge-v1.1.0 master-v110`), then read
where each hunk landed. `mmvq.cu`, `gated_delta_net.cuh`, `norm.cuh`, `fattn-tile.cuh`, `fattn-vec.cuh` and
`ggml/CMakeLists.txt` carry exactly the old lines; `common.cuh` differs in the three shuffle helpers; `ggml-cuda.cu` in
the few-row MMQ line and the replay exclusion; `norm.cu` and `gated_delta_net.cu` are the hand ports.

## Findings, most serious first

| # | Tree | What | On gfx906 |
|---|---|---|---|
| 1 | master | term 01: the rows-per-block change of `calc_rows_per_block` landed in upstream's new ORIN table; the GCN table is back to upstream's 2 rows (2..8 columns) and gets 1 row at 9..16 | yes, speed only |
| 2 | merge, master | the chunked gdn kernel does not write per-token snapshots: with K > 1 it writes slot 0 only and reads the input state at a K-wide stride | when the column-thread kernel declines |
| 3 | master | term 01 on K-quants: upstream's new VEGA20 rule sends Q4_K above 7 columns and NVFP4 above 6 to MMQ, so "16 columns" no longer holds for them; the fused norm still emits a Q8_1 row for those widths | yes, speed only |
| 4 | merge, master | the repacked MUL_MAT_ID path keeps the old ids helper, which cannot take duplicate expert ids; v1.1.0 made them legal | latent, no producer in the tree |
| 5 | merge, master | the MMA gdn launch is still not guarded for K > 1 (the old master had `!keep_rs`, the port has `!src_pend`) | no |

Everything else I was asked about reads right; section 6 lists it with what remains UNSURE.

---

## 1. master: term 01's rows per block are in the wrong table (silent, clean cherry-pick)

`/root/wt-master-v110/ggml/src/ggml-cuda/mmvq.cu`

Upstream added an ORIN branch to `calc_rows_per_block` (#31268) whose tail is the same text as the tail of the
GENERIC/GCN/TURING/GB10 branch (`case 6: case 7: case 8: return 2; default: return 1; } } return 1; }`). The hunk of
term 01 (`6226b3dba`, old `mmvq.cu` hunk `-613,7 +651,16`) applied there instead of in the GCN branch:

- `:700-723` the branch that GCN takes: `case 2..8: return 2;` (`:720`), `default: return 1;` (`:721`). Unchanged
  upstream code.
- `:725` `if (table_id == MMVQ_PARAMETERS_ORIN) {`, and inside it `:755`
  `return table_id == MMVQ_PARAMETERS_GCN ? (q8_lds ? 2 * nwarps : GGML_MMVQ_GCN_ROWS) : 2;` and `:764` the 9..16 case.
  `table_id` is ORIN here, so both conditions are dead.

Result on gfx906 (GCN table), all types that take `mmvq.cu`:

| Columns | Old master | master-v110 |
|---|---|---|
| 2..8 | `GGML_MMVQ_GCN_ROWS` = 4 (`:94-95`) | 2 |
| 9..16 | `GGML_MMVQ_GCN_ROWS_HI` = 2 (`:130-131`) | 1 (the `default`) |

Kernel and launcher both call the function, so the results are right; only the tuning is gone ("rows 4 wins up to 8
columns, rows 2 wins at 9-16", `:116`). At 9..16 columns it is worse than upstream's own 8-column setting: one row per
block re-reads every activation column for each row. `GGML_MMVQ_GCN_Q8_LDS_Y` (off by default) would hit its
`static_assert` at `:892` if someone turned it on. The one-column knobs (`:709-711`, term 05) and `calc_nwarps`
(`:549`, `:566`) are in the right place.

Which service path this touches: every weight type that goes through `mmvq.cu`; Q8_0 only when its weights are not in
the repack buffer type (repack is on by default, `common/arg.cpp:2406-2410`). I did not determine which of the two the
`-sm tensor` service takes for Q8_0.

Fix: move the two `return` lines and the 9..16 case labels from `:755-764` to `:713-722`, restore `return 2` in the
ORIN block.

Test: decode at 2..8 and at 9..16 parallel sequences on one die, master-v110 against the installed `-r2` deb, Q4_K
model (or Q8_0 with `-nr`): `llama-batched-bench -npl 4,8,12,16`. A compile-time check that would have caught it:
`static_assert(calc_rows_per_block(GGML_TYPE_Q8_0, 8, MMVQ_PARAMETERS_GCN) == GGML_MMVQ_GCN_ROWS)`.

## 2. merge and master: the chunked kernel with K > 1 (your question B)

The premise "the fork's chunked kernel with keep_rs writes per-token snapshots" is what the comment says
(`/root/wt-merge-v110/ggml/src/ggml-cuda/gated_delta_net.cu:300`) and not what the kernel does. This is his code, the
same in `557a6068f` (the merge did not touch `gated_delta_net_chunk.cu`), and it was in our PR #24 as well.

`/root/wt-merge-v110/ggml/src/ggml-cuda/gated_delta_net_chunk.cu`:

- `:242` `template <bool KDA, bool /*keep_rs_t*/>`: the launcher ignores the switch. The kernel (`:70-240`) has no
  snapshot code; it writes one state, at the end (`:228-239`), which is slot 0. The ring commit `7c5afc123` only added
  the explicit instantiations for `keep_rs_t = true` and the comment.
- `:107` `state_in_offset = sequence * K * H * S_v * S_v + ...`: the per-sequence stride of the INPUT state is K states.
  The op's state input is `[S_v, S_v, H, n_seqs]` (the recurrent kernel uses `sequence * H * S_v * S_v`,
  `gated_delta_net.cu:80`; the column-thread kernel too, `gated_delta_net_colthread.cu:101`). With K > 1 every sequence
  after the first reads past its state.
- The launcher has no `state_slot_stride` parameter, which does not matter only because it never writes another slot.

When a K > 1 call reaches it: `gated_delta_net.cu:318` takes it by default for `n_tokens >= 128` (32 for KDA) and
`S_v <= 128`, after the column-thread kernel declined (`:294`). That kernel declines
(`gated_delta_net_colthread.cu:458-466`) for `S_v != 128`, for q, k or state not 16-byte aligned, with
`GGML_GDN_NO_COLTHREAD=1`, and on every device whose warp is not 64 lanes. So:

- gfx906, S_v = 128 (the Qwen3.5/3.8 family), default settings: the column-thread kernel takes every call, it handles
  K and the slot stride correctly (`colthread.cu:251-253`, `:428`). Not reachable.
- gfx906 with a delta-net model of head size 16, 32 or 64, or `GGML_GDN_NO_COLTHREAD=1`: a ubatch of 128+ tokens in a
  ring context (or with `LLAMA_RS_REPLAY=0`) leaves slots 1..K-1 unwritten. The graph copies them into the ring
  (`src/models/delta-net-base.cpp`, `n_written = min(n_tokens, K)`) and the ring accepts a rollback into them. Same
  failure as the MMA case of the first report (silent wrong state after a rollback of 1..K-1 tokens into that ubatch),
  but on gfx906. With several sequences (`LLAMA_RS_REPLAY=0`, plane snapshots) the wrong input stride is added.
- NVIDIA and RDNA: every K > 1 call of 128+ tokens.

A run that shows it took this path prints `gdn-chunk-snapshots: keep_rs chunked path TAKEN (default)`
(`gated_delta_net.cu:312`). `LLAMA_GDN_NO_CHUNK_SNAPSHOTS=1` keeps K > 1 calls off the kernel (`:302-305`, `:318`).

Not covered by a test: the six restored snapshot cases have `n_tokens == K` (2 or 4,
`tests/test-backend-ops.cpp:14919-14925`). The CPU reference writes all `min(n_tokens, K)` slots
(`ggml/src/ggml-cpu/ops.cpp:11462-11464`).

Test on the box: add `test_gated_delta_net_snapshots(GGML_TYPE_F32, 4, 64, 200, 1, 1, false, 4)` (head size 64, 200
tokens, K 4) and a two-sequence one, run `test-backend-ops -o GATED_DELTA_NET`: expected failure on slots 1..3; with
`LLAMA_GDN_NO_CHUNK_SNAPSHOTS=1` expected pass. The existing S_v 128 cases plus `GGML_GDN_NO_COLTHREAD=1` do the same
for the head size the service uses.

Fix (small): `if (chunk_eligible && !keep_rs_t)` at `gated_delta_net.cu:318`, as it was before `7c5afc123`.

The rest of B, all from `/root/wt-merge-v110/ggml/src/ggml-cuda/`:

- Can a fused-cache call with `cache->in_dst` reach a fork kernel: no. `in_dst` is filled only for an op with replay
  sources and a scalar gate (`ggml-cuda.cu:4860`); `rep.st` is taken from it only under `if (src_pend)`
  (`gated_delta_net.cu:494-520`); such an op is launched with `REPLAY = true` on every path (`:553-554`, `:561-562`,
  and the replay-only launch `:544`), and `launch_gated_delta_net` reaches the fork kernels only under
  `if constexpr (!replay)` (`:291`). The MMA shortcut for replay ops is taken only without stores (`:535`).
- KDA with replay: the kernel has no stores for KDA (`:164`), and the fusion does not offer them (`ne[0] == 1` at
  `ggml-cuda.cu:4860`), so the four CPYs run as nodes. Consistent.
- K > 1 into code that assumes K == 1: the chunked kernel (above), the MMA launch (section 5), and nothing else. A
  replay op asserts `K == 1` (`gated_delta_net.cu:498`), and `keep_rs` is tested before `src_pend` in the dispatch
  (`:550-565`), so the two cannot mix.
- The replay sources are gathered copies (`build_rs`, `src/llama-graph.cpp:4437`), not views of the pending rows, so the
  kernel's stores (`gated_delta_net.cu:164-183`) do not write what another block of the same head is still replaying.

## 3. master: term 01 on Q4_K and NVFP4 after upstream #32805 (question D)

`/root/wt-master-v110/ggml/src/ggml-cuda/mmvq.cu:494-504`: upstream's new VEGA20 block in `ggml_cuda_should_use_mmvq`
returns `ne11 <= 7` for Q4_K and `ne11 <= 6` for NVFP4, everything else `ne11 <= MMVQ_MAX_BATCH_SIZE`. In the old
master gfx906 fell through to the last line of the function, so the raised macro (16, `mmvq.cuh:3`) applied to every
type. Now:

- Q8_0 and the other types: MMVQ up to 16 columns, as the commit says.
- Q4_K: MMVQ up to 7, MMQ from 8. NVFP4: MMVQ up to 6. Upstream measured this on an MI50 at the 8-column boundary; the
  Exabit measurement was MMVQ against MMQ at 9..16. For Flash-Next UD-Q4_K_XL the decode path at 8..16 sequences is a
  different kernel than in v1.0.0 master. Not wrong, but the commit message no longer describes it, and with finding 1
  the two cannot be told apart in one timing.
- Side effect on term 02: `ggml_cuda_rms_norm_q8_target` (`ggml-cuda.cu:5784-5823`) emits the Q8_1 row whenever a
  quantised MUL_MAT reads the norm and the batch is at most 16 rows (`:5794`, `:5801`); it does not ask
  `ggml_cuda_should_use_mmvq`. For Q4_K at 8..16 rows the consumer is now MMQ, which quantises for itself, so the row
  is computed and never read. Cost only (UNSURE how much).

Test: Q4_K model, 8 and 16 parallel sequences, once as built and once with the VEGA20 block's Q4_K line removed.

## 4. merge and master: the repacked MoE path and duplicate expert ids

Upstream #32505 (`b94959a64`) rewrote the ids helper in `mmid.cu` so that one token may name the same expert twice,
removed the "undefined behaviour" note from `ggml.h`, and enabled the `test_mul_mat_id_dup` cases. The merge takes
upstream's `mmid.cu` whole (no fork delta). The repacked path still calls the fork's copy of the OLD helper:
`/root/wt-merge-v110/ggml/src/ggml-cuda/q8_repack/mul-mat-id.cu:56`, `:536` ->
`q8_repack/mmid-bounds.cuh`. There, two slots of one token that name the same expert write one store entry
(`mmid-bounds.cuh:85`, `:97-99`) while every higher expert counts both (`:82`). The compact layout then has a column
that no expert filled but that lies inside an expert's bounds: its `ids_src1` / `ids_dst` entries are whatever the pool
held, and the kernels index src1 and dst with them.

Reachable today: no graph in the tree produces duplicate ids, and `test-backend-ops` allocates its weights in the plain
buffer type, so its new duplicate cases run upstream's helper and pass. It becomes live with the first model that
routes with repetition on a Q8_0 (or other repacked) MoE. UNSURE whether such a model is planned; the contract changed,
so I list it.

Test: none with the current tools (the repack path needs a repack buffer). A unit test that puts the expert tensor of
`test_mul_mat_id_dup` into the repack buffer type would show it.

## 5. merge and master: MMA launch with K > 1 (known, for completeness)

`/root/wt-merge-v110/ggml/src/ggml-cuda/gated_delta_net.cu:529` `if (!src_pend)` and
`/root/wt-master-v110/ggml/src/ggml-cuda/gated_delta_net.cu:631` `if (!src_pend && !pf.folded())`. The old master had
`if (!keep_rs && !pf.folded())`; the port follows his tree, where the guard never existed. Section 3 of the first report
applies unchanged (NVIDIA GA10x/Ada/Blackwell/GB10, CDNA, RDNA3.5; not gfx906). One word in each line fixes it.

## 6. Read and found right

Merge, files where git reported no conflict:

- `ggml-cuda.cu`: the gdn cache fusion with the pending stores (`:4797-4900`) is upstream's text and meets no fork
  code. NORM + MUL (+ADD) (`:5551-5569`, `:6608-6616`) and the MROPE extension (`:4768-4781`) sit in front of the RMS_NORM
  fusions and do not overlap the fork's. The duplicate-id change is in the host fallback only (`:3670-3671`). The
  MUL_MAT_ID dispatch has upstream's few-row MMQ (a stub that returns false on HIP).
- `gated_delta_net.cu`: section 2.
- `mmvq.cu`: the fork's two deltas (Q8_0 five-column warps, the Q8_1 cache) are untouched by upstream's 94 lines. The
  RDNA1 and ORIN tables are separate branches, the VEGA20 rule is section 3.
- `fattn-common.cuh`: upstream's #32901 does not touch the lines around the NaN guard (`:1282-1285`) or the
  parallel-blocks cap (`:1545-1585`); `parallel_blocks` is not used between the cap and `blocks_num.y`. The indexed
  bias (#32740) uses sources 6 and 7 and op param 5, which the fork does not use, and CUDA and the meta backend both
  refuse it.
- `common.cuh`, `mmq.cuh`, `quantize.cu`, `ggml-alloc.c`, `ggml-backend-impl.h`: mechanical. The caps initialisers
  match the struct order in both backends (`ggml-backend-meta.cpp:148-155`, `ggml-cuda.cu:7759-7766`). #32374 widened
  the dst offset of upstream's MMQ kernels to 64 bits; whether the repack GEMM kernels have the same 32-bit limit above
  2^31 dst elements I did not check (UNSURE).
- `ggml-backend-meta.cpp`: the GATED_DELTA_NET rule is upstream's with the fork's comment
  (`:1291-1318`): pend_k, pend_v, pend_gb by heads (axis 1), n_rep mirrored, asserted. The split of the pending tensors
  themselves comes from upstream's rules in `src/llama-model.cpp:1010-1016`, `:1082-1090`, which merged in. The stores
  are ordinary CPY nodes that every lane runs on its slice; the CUDA fusion that absorbs them is decided per lane on the
  lane's graph, and when a subgraph boundary separates the op from its CPYs the fusion does not match and the nodes
  run. I found no fork rule that looks at only six sources. UNSURE: I did not trace a view or reshape of a repeated
  segment (`nr > 1`) through the fork's own handlers; your `rs-modes` run went through them without an assert.
- `ggml-backend.cpp`: nothing upstream relies on is missing. No backend in the tree calls `add_scratch`, so the
  scratch tensors are inert (and the meta backend does not forward `graph_optimize` parameters at all).
  `fast_async_host_input` is stored (`:3143`) and never read, which is the intended state. The property of the new test
  (an input may be overwritten when `compute_async` returns, `tests/test-backend-sched.cpp:839-934`) holds: a user input
  is either copied into the staging buffer by `memcpy` or copied synchronously after a wait, so the host memory is read
  before the call returns. It holds by blocking, so the second compute of that test waits for the sleep.

Master, the ports:

- terms 02 and 03, `norm.cu`: the kernel arithmetic is the old one (`:290-313` vec, `:353-365` scalar; the standalone
  `quantize_q8_1` it mirrors did not change upstream). All eight combinations of clamped, q8 and pre-add exist for float
  rows with and without the bias add (`:753-790`). For BF16 rows the launcher instantiates no q8 or pre-add kernel
  (`if constexpr` at `:759`, and `static_assert` at `:188-189`), and the predicates only offer float tensors
  (`ggml-cuda.cu:5793`, `:5841-5842`). The pre-add branch reassigns `xc`, which compiles only for float and is
  discarded otherwise (`:318-324`). The width-8 and width-32 reductions stay inside one half of a 64-lane wave.
- terms 04, 07, 08, 10, 12: no other path reads the graph's q, k, g or beta of a folded op. The MMA launch excludes a
  folded call (`gated_delta_net.cu:631`), the replay-only launch exists only for replay ops, which are never folded
  (`ggml-cuda.cu:5960-5964`, asserted at `gated_delta_net.cu:622`), and the pending stores exist only for replay ops.
  The fork kernels are skipped for a folded call (`:363`, `:375`). The fold hook runs before `try_fuse`, so upstream's
  new RMS_NORM fusions cannot take a producer first.
- terms 23 and 28, `common.cuh`: same text as the old commit except `ggml_cuda_shfl_xor` at the 64-lane step
  (`:606`, `:618`, `:630`). New upstream callers since v1.0.0: `mmid.cu` and `topk-moe.cu` at width 64 with every lane
  active, the gdn replay loop at 16. A width that is not a power of two would stop the build at the `static_assert`.
  `GGML_GCN_NO_DPP` still precedes `common.cuh` in every flash-attention instance file.
- terms 15, 05, 09 and the two all-reduce commits: in place.
- CMake: `GGML_CUDA_FA_QUANTS=all` is a value upstream's new `ggml_cuda_fattn_instances()` knows
  (`ggml/cmake/common.cmake:73`). The LLVM 24 rule's glob matches the head-256 instance of every K/V type, which is the
  same set of kernels the single v1.0.0 file held.

## 7. For the open `-np 4` question (replay against plane snapshots)

Replay ops run on the recurrent kernel, snapshot ops on the column-thread kernel. To hold the kernel fixed without a
rebuild: `GGML_GDN_NO_COLTHREAD=1 LLAMA_GDN_NO_CHUNK_SNAPSHOTS=1` puts the snapshot arm on the recurrent kernel too
(the second switch is needed because of section 2). If the two arms then agree, the difference was the kernel; if
not, it is in the snapshot bookkeeping for several sequences.
