# A - ggml core and the CUDA/HIP backend: his b12500 merge against ours

Scope: `ggml/src/ggml-backend.cpp`, `ggml-backend-meta.cpp`, `ggml/src/ggml-cuda/*` as listed in the task, his commits
`9284233fe`, `0c3567111`, `773721de3`.

Method: reading only. Nothing was built and no GPU was used, so every statement below about run-time behaviour is
derived from the code, not observed. `file:line` is his tree at `557a6068f` unless it says "ours" (= the scratch
worktree `/root/wt-cmp-ours-b12500`) or names another commit. Two things were executed, both read-only: the ROCm
compiler's predefined macros for gfx906 (`__GFX9__` is defined, so the physical warp size is 64 in a gfx906 build), and a
3-way merge of the 20 Exabit commits onto his tree with `git merge-tree` writing into a scratch object directory
(section 12).

## Verdicts at a glance

| # | Topic | Verdict | Reachable on gfx906 |
|---|---|---|---|
| 1 | `topk-moe.cu` fused softmax reduces 32 of 64 lanes | defect in HIS tree, and OURS carried to b12500 has the same lines (clean merge, both sides) | yes |
| 2 | `topk-moe.cu` expert-group ballot (our `7334d8c27`) | UPSTREAM-MOVED: fixed by upstream #31617, drop ours | - |
| 3 | `gated_delta_net.cu` MMA kernel with the snapshot ring | OURS-FIXES-HIS | no (NVIDIA, CDNA, RDNA3.5 only) |
| 4 | `ggml-backend-meta.cpp` `same_lanes` (`773721de3`) | HIS-BETTER | only with a zero-sized slice (test fixtures) |
| 5 | `ggml-backend.cpp` input prefetch | HIS-BETTER (he left it out; our port has an abort path under `-sm tensor`); reachability UNSURE | UNSURE |
| 6 | `ggml-backend.cpp` `hv_tensor_requires_copy` / unread inputs | SAME behaviour (his field is dead storage) | - |
| 7 | `9284233fe` shuffle helpers in fork kernels | UPSTREAM-MOVED, his commit is required: ours carried to b12500 does not compile on HIP | yes (build) |
| 8 | `pad_reflect_1d.cu` (our `a73c601e6` vs upstream #32179) | UPSTREAM-MOVED, EQUIVALENT for the HIP limit: drop ours | - |
| 9 | `ggml-cuda.cu` graph executables, out-of-memory fallback | EQUIVALENT (two small edges for ours, no failure shown) | - |
| 10 | `mmid.cu` / `mmid.cuh` / `q8_repack/mmid-bounds.cuh` / `q8_repack/mul-mat-id.cu` (`0c3567111`) | EQUIVALENT | - |
| 11 | `norm.cu` clamped-grid plumbing | EQUIVALENT | - |
| 11 | `mmq.cu`, `common.cuh`, `repack-common.cuh` (rest), `fattn-vec.cuh`, `top-k.cu`, meta comments | SAME | - |

Counts: 1 defect in both, 1 OURS-FIXES-HIS, 2 HIS-BETTER, 3 UPSTREAM-MOVED, 3 EQUIVALENT, 7 SAME.

---

## 1. topk-moe.cu: the fused softmax reduces only half of a 64-lane row (his tree; ours carried has it too)

Not one of the brief's labels: it is a defect in his tree that our merge does not fix. It is the most important
finding of this group because it is on gfx906 and it came out of a clean merge.

What happened. The fork's `b549a9862` ("fused top-k MoE router sums in the unfused order") rewrote
`softmax_warp_inplace` with two-stage reductions written for 32-lane rows: `warp_reduce_max(...)` and
`warp_reduce_sum(...)` with the default template width `WARP_SIZE` = 32 (`common.cuh:46`, `:522`, `:582`). Upstream
#31617 (`278a89cb7`, in b12500, not in b12359) then made a row one physical warp: `block_dims(warp_size, rows_per_block)`
(`topk-moe.cu:403-405`) and `warp_size = ggml_cuda_get_physical_warp_size()` = 64 on GFX9 (`common.cuh:419-425`). Git
merged the two without a conflict in this function, on his side and in our scratch worktree alike (ours
`topk-moe.cu:18-70` is identical to his; the conflict markers in our worktree are at 140, 361 and 420 only).

His code now (`topk-moe.cu`):

- `:30` `idx = lane + i * warp_size`: with `warp_size` 64, lane L holds expert `i*64 + L`, lanes 32..63 included.
- `:32` `group_reductions[i] = warp_reduce_max(active ? vals[i] : -INFINITY);` width 32. On HIP a width-32 shuffle
  stays inside each 32-lane half (`/opt/rocm/include/hip/amd_detail/amd_warp_functions.h:521-524`), so lanes 0..31 get
  the max of experts `i*64 + 0..31` and lanes 32..63 the max of `i*64 + 32..63`.
- `:38-41` `if (lane == i) max_val = group_reductions[i];` picks lanes 0..experts_per_thread-1, all in the low half. The
  result of the high half is never used.
- `:44` `warp_reduce_max<warp_size>(max_val)`: 64 wide, but over values that only cover the low halves.
- `:56`, `:62-65`, `:68`: the same for the sum.

Result on a 64-wide wavefront: the softmax maximum and denominator leave out every expert whose index mod 64 is 32 or
more. Every weight is too large by the factor (full sum / half sum). The selected ids do not change (one common scale).
`n_experts` <= 32 is not affected (all experts are in lanes 0..31). The delayed softmax over the selected weights is not
affected for `n_expert_used` <= 32. The `with_norm` block was a conflict and he ported it correctly to
`warp_reduce_sum<warp_size>` (`topk-moe.cu:354`, `:366`), which is why only the softmax is wrong.

Failure scenario. A HIP build whose physical warp size is 64 (gfx906 is one), a MoE router with softmax gating before the
top-k, an expert count that the fused kernel accepts and that is above 32: 64, 128, 256, 288, 384, 512, 576
(`topk-moe.cu:524`), any batch size (the fusion is taken whenever the pattern and the memory-range check match,
`ggml-cuda.cu:5661-5732`).
- Without weight normalisation (`norm_w` false: OLMoE `src/models/olmoe.cpp:134`, DeepSeek-MoE v1
  `src/models/deepseek.cpp:146`, both 64 experts): the expert weights are wrong by a factor near 2, the output is wrong.
- With weight normalisation (`norm_w` true: `qwen3moe.cpp:136`, `qwen35moe.cpp:479`, `qwen3next.cpp:561`, the
  Flash-Next family): the factor cancels in the division by the sum of the selected weights, so the result is right to
  rounding. What is lost is the property `b549a9862` was written for: fused and unfused are no longer bit-equal, so a
  speculative verify batch can differ from plain decode again.
- DeepSeek-V4.1 (sqrt-softplus / sigmoid gating) and the grouped DeepSeek-V3 path do not use this function.
- Qwen3.8-27B is not a MoE: not affected.

Why I believe it and what I did not check: the three facts (default width 32, 64-lane rows on GFX9, width-32 shuffles
stay in a half) are each read from the source and the ROCm header. I did not run it. If `test-backend-ops -o TOPK_MOE`
passes on his build on gfx906, this section is wrong and I would want to know why.

Test on the box (one die is enough): build `557a6068f`, `test-backend-ops -o TOPK_MOE -b ROCm0`. Expected by this
reading: failures in the softmax cases without norm for `{128,1}`, `{256,22}`, `{288,22}`, `{384,*}`, `{512,3}`
(`tests/test-backend-ops.cpp:13427-13450`), none for 8, 31, 32 (not above 32) and none for 40, 71, 129, 160 (not
fused). Model level: any softmax MoE with `GGML_CUDA_DISABLE_FUSION=1` against without (`ggml-cuda.cu:223`).

Fix (small, goes into the PR): the two inner reductions at `topk-moe.cu:32` and `:56` must cover the row. Width
`warp_size` makes the values right. Whether that also restores bit-equality with the unfused softmax on a 64-lane row is
a separate question (the unfused order groups 32 values); his acceptance for `b549a9862` (same text and top-5 with MTP
verify on Flash-Next) is the test for that.

## 2. topk-moe.cu: the expert-group ballot on 64-wide wavefronts (question e, first half)

Answer: his tree contains no fork-side fix and needs none at b12500. Upstream fixed it structurally in #31617
(`278a89cb7`), which is in b12500 and not in b12359 or b12169.

- His tree between the two merges (`773721de3:ggml/src/ggml-cuda/topk-moe.cu:135`) still had upstream's
  `__ballot_sync(0xFFFFFFFFULL, ...)` on 32-lane rows, that is the defect our `7334d8c27` fixed. He never carried a fix.
- At `557a6068f` a row is one whole wavefront: `block_dims(warp_size, rows_per_block)` with the device's warp size
  (`topk-moe.cu:403-405`, `ggml-cuda.cu:396`), lane = `threadIdx.x` in 0..63, and the ballot is
  `ggml_cuda_ballot(...)` (`topk-moe.cu:140`) = `__ballot_sync(0xffffffffffffffff, pred)` with a 64-bit mask type on HIP
  (`common.cuh:130`, `:455-457`, `vendors/hip.h:15`), which HIP evaluates as `__ballot(pred) & mask`
  (`amd_warp_sync_functions.h:186-193`). One row, one ballot, 64 bits: the two causes of our bug (two rows in one
  wavefront, a 32-bit mask ANDed onto the result) are both gone.
- Verdict UPSTREAM-MOVED. Drop `7334d8c27` when carrying to b12500. (If its `#if defined(GGML_USE_HIP)` block were kept
  it would compute `row_lane0` = 0 and still be right, just redundant.)
- Not run. Test: `test-backend-ops -o TOPK_MOE` cases with `n_expert_groups` > 1 and more than one row
  (`tests/test-backend-ops.cpp:13446-13450`); section 1 makes other cases of the same run fail, so read the groups cases
  separately.

Same file, third fork delta: `rows_per_block = 1` for one row on Vega 20 (`topk-moe.cu:401-402`): SAME on both sides.

## 3. gated_delta_net.cu: upstream's MMA kernel with the snapshot ring (question d)

- His: launches `ggml_cuda_gdn_mma_launch` unconditionally (`gated_delta_net.cu:392-398`).
- Ours: `if (!keep_rs)` in front of it (ours `gated_delta_net.cu:392-393`), from our v0.7.0/v0.9.0 merges.
- Verdict OURS-FIXES-HIS. Not reachable on gfx906.

Where the MMA path is reachable: `select_gdn_mma_path` (`gated-delta-net-mma.cu:437-486`). On HIP it returns a kernel
only for RDNA3.5 with at least 2048 tokens (`:445-449`) and for CDNA with MFMA (`:451-462`), else `gdn_path::ar`
(`:463`). Vega 20 is `0x906`, below `GGML_CUDA_CC_CDNA1` = `0x908` (`common.cuh:72`, `:75`, `:99`) and not RDNA, so on
gfx906 the launch always returns false and both trees run the same fork kernels. On CUDA it is taken on GA10x, Ada,
Blackwell and DGX Spark (`:469-471`). Eligibility on any of them (`gated_delta_net.cu:312`,
`gated-delta-net-mma.cu:439`): scalar gate, `K < n_tokens`, S_v = 128, key head count 16, H in {16,32,48,64}, at
least 64 tokens. The CDNA branch is new in b12500 (#31327 `47fe13331`), so the exposure grew with this merge.

What goes wrong in his tree there:
- The MMA kernel writes one state, the final one (`gated-delta-net-mma.cu:401`). Upstream allows that: "when n_tokens >
  K only slot 0 is guaranteed" (`ggml/include/ggml.h:2423`), and upstream's memory then refuses a rollback
  (`cell.rs_max = ... : 0`, `src/llama-memory-recurrent.cpp:885`).
- The fork's ring does not follow that contract. The graph copies `n_written = min(n_tokens, K)` slots of the op output
  into the ring (`src/models/delta-net-base.cpp:156-169`), the ring counts all of them as valid
  (`llama-memory-recurrent.cpp:598-601`) and accepts a rollback of up to `n_rs_seq` tokens
  (`llama-memory-recurrent.cpp:285-288`). His own `1c8f7fa09` says "the ring needs every snapshot slot written by the
  delta-net kernels" and enables the ring for "CUDA" and "ROCm".
- So: ring on (`n_rs_seq` > 0, one sequence), a ubatch of 64 or more tokens on one of those GPUs, then a rollback of
  1..K-1 tokens into it (a rejected draft, a regenerate, the server re-evaluating the last prompt tokens): the restored
  recurrent state is whatever was in slots 1..K-1 of the compute buffer. Silent, wrong continuation.
- His tests cannot see it: the dummy models do not have S_v = 128 with 16 key heads, so the MMA path is never eligible in
  `test-llama-context`.

Cost of ours: `keep_rs` is `K > 1`, so we also skip the MMA kernel when K > 1 without the ring (several sequences),
where upstream's contract would allow it. That is a speed loss on those GPUs only, never on gfx906.

Test on the gfx906 box: none, the path cannot be entered. On an RTX 30/40/50: `test-llama-context` rollback cases with
a fixture of S_v 128 / 16 key heads and a prefill ubatch of 64+ tokens.

Collision: the Exabit GDN folds already carry `if (!keep_rs && !pf.folded())` at this line; see section 12.

## 4. ggml-backend-meta.cpp: `same_lanes` (his `773721de3`) (question c)

- His (`ggml-backend-meta.cpp:4268-4279`): the reduce of a PARTIAL node is folded into the following ADD of two partial
  sums only if both operands carry the compute flag on the same lanes.
- Ours (ours `ggml-backend-meta.cpp:4265-4269`): folds always (`i_delayed = k`). Our branch forked from `cdc81dae4`,
  before this fix existed; it is his later development, not a merge decision of ours.
- Verdict HIS-BETTER. Take his commit as it is.

The failure it prevents: a lane whose slice of a source is zero-sized gets the compute flag cleared on that node
(`ggml-backend-meta.cpp:1856-1874`). When the AllReduce of node `i` is delayed to `i_delayed`, the zero-slice rule
clears the flag on that lane for every node from `i+1` to `i_delayed` (`:4291-4300`). The fold (`:4251-4285`) moves
`i_delayed` to the ADD, and the nodes in between are the producer chain of the other operand, the shared expert. So a
lane with a zero slice of the routed experts and a non-zero slice of the shared expert has its share of the shared
expert switched off and contributes zeros. His message: `test-llama-archs` glm5-next, 2-GPU tensor row, NMSE 3.52e-04
(the test fails above 1e-4, `tests/test-llama-archs.cpp:1460`) against 8.75e-07 with the fix.

Does our merge have it: yes, the code is the pre-fix code. Reachable only when a lane has a zero-sized slice in one
operand and not in the other. Slices are `ne * (k+1) / n_lanes` rounded down to the granularity
(`src/llama-model.cpp:1386-1403`), so that needs a dimension smaller than lanes x granularity: the tiny fixtures of
`test-llama-archs`, not the models served on the box at 4 lanes. A `--tensor-split` with a zero weight gives a zero
slice in both operands, which is the same-lanes case and unaffected. Our 2026-11-14 report has `test-llama-archs`
passing on four dies; his failure was at two.

Test on the box: `HIP_VISIBLE_DEVICES=0,1 test-llama-archs`, "Meta" row of glm5-next, on our tree and on his.

Other differences in this file, both SAME: the comment at `:2602` (his still names the V4.1 host Engram gather, ours
generalised the wording) and a five-line comment ours adds above the cgraph re-creation loop (ours `:4482-4486`).

## 5. ggml-backend.cpp: upstream's input prefetch (question b, first half)

What his scheduler does with it: nothing. Upstream's `ggml_backend_sched_prefetch_inputs` and the "wait for the copy
only" shortcut are not in his file. The state they would use is kept in the struct with the comment "unused: the
fork's split-input loop replaces upstream's input prefetch, kept pending the prefetch A/B"
(`ggml-backend.cpp:1321-1328`), and the events are freed in `ggml_backend_sched_free` (`:3110`, always NULL). His copy
loop is the fork's loop unchanged (`:2679-2950`).

What ours does: ports the prefetch into that loop (ours `ggml-backend.cpp:2632-2710`, called between the two passes at
`:2752-2757`, the wait at `:2767-2769`), with the extra rule that an input no computed node reads is not prefetched
(`:2668`).

Can either side skip a needed copy or wait on a copy that never happens:
- His: no wait exists. A copy is skipped only by the `inputs_read` rule (section 6), which is identical on both sides.
- Ours: a wait happens only for a tensor in `sched->prefetched`, which is filled only after
  `ggml_backend_tensor_get_async` was issued for it, and the event is recorded after the copies (ours `:2701-2707`). If
  the backend has no events the function returns before issuing anything (`:2696-2698`). `n_prefetched` is reset at
  every split (`:2650`). The skipped normal copy would have copied the same tensor, which the current split does not
  write (`:2680-2690`). So in upstream's model: no. I found no ordering hole on HIP either: the fork reads host memory
  synchronously on every path into a GPU or meta split (staging memcpy `ggml-backend.cpp:2723`, `:2918`; the sync
  fallback `:2942`; `cudaMemcpy` for host sources on HIP `ggml-cuda.cu:4160-4161`), and the prefetch copies run on the
  main stream of each lane (`ggml-cuda.cu:4059-4066`, `:4113-4122`) after the meta backend has joined its dispatch
  workers (`ggml-backend-meta.cpp:5334-5360`).

What ours adds that his does not have, and why I rate his as the more robust state:
1. Under `-sm tensor` the prefetch is live: the meta backend has `get_tensor_async` and events
   (`ggml-backend-meta.cpp:5473`, `:171-188`). The async getter is stricter than the synchronous one that the normal
   path uses (`ggml_backend_tensor_copy` at `ggml-backend.cpp:2942`): it asserts a contiguous tensor and one unrepeated
   segment (`ggml-backend-meta.cpp:3380`, `:3383-3384`), while the synchronous getter handles several segments,
   repeats and a non-contiguous MIRRORED tensor (`:2188-2249`). An input with such a layout is copied fine by his
   scheduler and aborts in ours.
2. It makes correctness depend on an event recorded after an async device-to-host copy on ROCm. The fork documents two
   places where an event did not order a copy there (`ggml-backend.cpp:2769-2776`, `ggml-cuda.cu:4271-4279`). Those are
   stream-side waits and peer copies, not this pattern, so this is a reason for the A/B he asks for, not a defect I can
   show.

Reachability (UNSURE): the prefetch fires only when a host split follows a GPU/meta split and reads a tensor that an
earlier split of that same backend computed and that was not copied to the host before. That needs at least two
GPU/meta splits with host splits between them. A fully offloaded graph is one meta split (plus the input layer), where
nothing is prefetched and both trees behave the same. His tree has an "experimental host Engram inputs" mode for V4.1
(`src/models/deepseek41.cpp:338-356`) that produces that shape. What I would need to know: the split list of each
served model, `GGML_SCHED_DEBUG=2` prints it.

For the PR: nothing from ours. If the prefetch is wanted later, it needs a layout guard for meta tensors (or the meta
async getter brought level with the synchronous one) and his A/B.

Test on the box: `GGML_SCHED_DEBUG=2` on each served model to count splits. A/B only if a host split sits between two
meta splits.

## 6. ggml-backend.cpp: `hv_tensor_requires_copy` and unread inputs (question b, second half)

- His: keeps upstream's field. It is allocated, cleared and freed (`ggml-backend.cpp:3055`, `:3142`, `:3124`), written in
  the split pass (`:2278-2282`), and never read: the loop that reads it in upstream (`aaa807c56`
  `ggml-backend.cpp:2238`, `:2247`) is replaced by the fork's loop. One byte per hash slot, no effect.
- Ours: drops the field.
- Both: the fork's `inputs_read` decides, set by `ggml_backend_sched_mark_read_inputs`
  (`ggml-backend.cpp:1845-1898`, called at `:2489`) and tested at `:2687`:
  `if (pass == 0 && input != input_cpy && !split->inputs_read[input_id]) continue;`. The function is byte-identical on
  both sides.
- Verdict SAME behaviour.

Differences of the fork's rule from upstream's, the same in both trees: a graph without compute flags keeps every copy
(`:1892`), and host weights (pass 1) are always copied, read or not. The second is the fork's choice (`4d03828a1`:
"host weights are always copied because the expert copy callback depends on them"); upstream skips them too.

## 7. `9284233fe`: shuffle helpers in the fork's kernels

Upstream #31617 removed the HIP macros that mapped `__shfl*_sync(mask, ...)` to the unmasked forms and turned on ROCm's
own sync builtins (`git diff b1b770d aaa807c56 -- ggml/src/ggml-cuda/vendors/hip.h`). Those reject a mask that is not
64 bits wide with a `static_assert` (`amd_warp_sync_functions.h:306-309`). The fork's own kernels still passed
`0xffffffff`.

- His: `9284233fe` switches them to `ggml_cuda_shfl`, `ggml_cuda_shfl_up`, `ggml_cuda_shfl_xor`:
  `gated_delta_net_chunk.cu:49`, `gated_delta_net_colthread.cu:46`, `q8_repack/mmid-bounds.cuh:91` and `:102`,
  `q8_repack/repack-common.cuh:239`. Same lanes, same widths: no change in behaviour.
- Ours carried to b12500: still has the raw calls at ours `gated_delta_net_chunk.cu:49`,
  `gated_delta_net_colthread.cu:46`, `q8_repack/repack-common.cuh:239`, so it does not compile for HIP (by the
  header's `static_assert` and his commit message; I did not build it). At b12359 they were right.
- Verdict UPSTREAM-MOVED. His commit is required, take it. No raw 32-bit mask is left in his `ggml-cuda/` (checked by
  grep). The Exabit DPP reductions bring three more, see section 12.

## 8. pad_reflect_1d.cu (question e, second half)

- His: no fork delta at all, the file is upstream's (same blob `6bad7d379` at `557a6068f` and `aaa807c56`), with
  upstream's #32179 (`2d8a32502`).
- Ours: `a73c601e6`, rows launched in chunks on HIP only, at most 2^31 - 1 threads along grid.x per launch.
- Upstream: at most `UINT32_MAX / 256` = 16,777,215 blocks of 256 threads per launch, that is 2^32 - 256 threads, the
  chunk offset passed to the kernel (`pad_reflect_1d.cu:22`, `:26`, `:82`, `:91-92`), on every backend.

Equivalent for HIP's limit: yes. The limit we measured on this box (in the message of `a73c601e6`) is "accepted with
2^32 - 1 threads along one grid dimension, rejected with 2^32", for blocks of 1 and of 2. Upstream's largest launch is
2^32 - 256, inside it; ours is 2^31 - 1, also inside it. Not measured: a launch of exactly 16,777,215 blocks of 256;
upstream's commit says its ROCm CI jobs pass with it.

Differences that do not matter in practice: upstream also splits a single row wider than the limit and asserts at most
`UINT32_MAX` blocks in total (`:85`); ours has neither, and a tensor that large (16 GiB of F32 in the source alone) does
not fit a 32 GB die together with its result. Upstream's version also covers CUDA's limit of 2^31 - 1 blocks, ours
leaves CUDA at one launch.

Verdict UPSTREAM-MOVED: drop `a73c601e6`, resolve our worktree's conflict to upstream's file.
Test: `test-backend-ops -o PAD_REFLECT_1D` on his build, 7 of 7 expected, including `ne_a=[2,16777217,1,1]`.

## 9. ggml-cuda.cu: graph executables and the out-of-memory fallback

The whole difference in this file (about 200 lines) is one feature written twice: the fallible instantiate, the HIP
kernel-set check before `hipGraphExecUpdate`, and what happens when the executable of a chunk does not fit. Plus the
order of two `#include` lines (`ggml-cuda.cu:36-37`), SAME.

His (`ggml-cuda.cu:4458-4596`, `:6492-6535`, `:6629-6638`, `:6741-6746`): `ggml_cuda_graph_try_instantiate` takes the
instance and also refreshes its clusters; the kernel functions of a chunk are collected inside
`ggml_cuda_graph_update_executable`; on a chunk that does not fit, `abandon_capture()` drops every executable, resets
the Q8_1 cache, sets `first_node = chunk_beg`, and the loop goes back with `i = chunk_beg - 1`.

Ours (ours `ggml-cuda.cu:4426-4634`, `:6541-6572`, `:6668-6676`, `:6779-6789`): the same steps with a helper for the
kernel list, a helper to drop the executables, `break` plus a second pass of the outer loop instead of rewinding `i`.

Verdict EQUIVALENT. Both end the capture before the failure is seen, both re-run the chunk from its first node
directly, both drop the Q8_1 entries the abandoned capture prepared, both release the capture lock once, both leave the
earlier chunks as launched. Three places where ours does a little more; none gives a failure I can construct with
default settings:

1. Kernel list at the first instantiate. Ours records it (ours `:4607`). His leaves `kernel_funcs[idx]` empty
   after the first instantiate (`:4584-4588`), so the first update of every chunk sees "kernel set changed" and
   instantiates again (`:4500-4507`). Safe, one extra instantiate per chunk and graph key. Not measured.
2. Shrinking graphs. Ours trims `kernel_funcs` with `instances` (ours `:6796`); his trims `instances` only (`:6752`). A
   stale list could make a later comparison come out "unchanged" against an executable built from another capture. It
   needs the chunk count of one graph key to shrink and grow again with the old kernel set coming back: theoretical.
3. `prev_i` on the re-run. Ours restores it (`first_prev_i`, ours `:6542-6543`, `:6598`); his re-runs with the stale
   value (mid-graph) or 0 (last chunk, `:6560`). It only decides whether a concurrent-stream region is launched at the
   first re-run node, and those regions exist only with `GGML_CUDA_GRAPH_OPT` set.

For the PR: nothing needed. Points 1 and 2 are two lines each if he wants them.
Test: `GGML_CUDA_GRAPH_CHUNK_SIZE` set small (`ggml-cuda.cu:4566`) on a die with almost no free memory forces the
fallback; perplexity must equal the run with graphs off.

## 10. mmid.cu, mmid.cuh, q8_repack/mmid-bounds.cuh, q8_repack/mul-mat-id.cu (his `0c3567111`)

Upstream #31047 changed `ggml_cuda_launch_mm_ids_helper` to write tile specs, and the repacked Q8_0 MoE kernels still
need the per-expert bounds.

- His: a copy of the old helper under its own name in `q8_repack/mmid-bounds.cuh` (183 lines), called at
  `q8_repack/mul-mat-id.cu:56` and `:536`. Upstream's `mmid.cu` is untouched.
- Ours: an optional `expert_bounds` output on upstream's helper (ours `mmid.cu:40`, `:126-135`, `mmid.cuh:12-17`), the
  repack calls pass `tile_spec = nullptr` and tile width 1 (ours `q8_repack/mul-mat-id.cu:55-57`, `:535-537`).
- Verdict EQUIVALENT. Up to the last statement the two kernels are the same text (his `mmid-bounds.cuh:47-123` against
  ours `mmid.cu:48-124`), and both write `expert_bounds[expert] = nex_prev` and, for the last expert,
  `expert_bounds[n] = nex_prev + it_compact` (his `:125-131`, ours `:127-131`). Ours returns before the tile code when
  `tile_spec` is null, so the tile width is never used.
- Trade-off: his copy does not conflict with upstream's helper but has to follow it by hand (it already needed
  `9284233fe`); ours has one kernel but edits upstream's signature. Keep his, nothing to send.
- Test: `test-backend-ops -o MUL_MAT_ID` with Q8_0 on a repack build, or DeepSeek-V4.1 Q8_0 prefill.

## 11. The rest: EQUIVALENT or SAME

- `norm.cu` (conflicted in our worktree): the fork delta is the `clamped` template parameter of `norm_f32`,
  `rms_norm_f32`, `l2_norm_f32`. Upstream's change in between is BF16 for RMS_NORM (#32135). His: `clamped` is a
  template parameter of `rms_norm_f32_dispatch` chosen by the callers (`norm.cu:487-488`, `:546-561`, `:594-640`,
  passed on at `:479`); every call of the dispatcher passes it (`:551`, `:608`, `:620`, `:627`). Ours at b12359: the
  choice is inside the dispatcher's launch macro. Same kernels instantiated, same condition
  (`nchannels > UINT16_MAX || nsamples > UINT16_MAX`). EQUIVALENT. Take his file; it is the one that already has
  upstream's `T` / `T_add`.
- `mmq.cu`: order and names of the trailing parameters of `ggml_cuda_mmq_get_J_best` (`mmq.cu:135-137`; his has
  `nsm > 0 &&` in the gfx906 rule at `:155`, ours does not), `ntj_chunk` against `ntj`. The two callers (`:273`,
  `:661`) pass the same values, and with the defaults of the second caller the rule is inactive on both sides. SAME.
  Shared and unchanged: `ggml_cuda_mmq_fusion_is_efficient` (`:661`) says "same tile width as ggml_cuda_mul_mat_q" but
  does not pass the gfx906 arguments, in both trees.
- `common.cuh`: position of the `streams` member in the context struct (`common.cuh:1441`). SAME.
- `q8_repack/repack-common.cuh`: besides section 7, a comment and a `GGML_UNUSED(src0)`. SAME.
- `fattn-vec.cuh` (conflicted in our worktree): the fork delta is the mask row stride `s31` (`fattn-vec.cuh:119`,
  `:308`), the same on both sides; the conflict is upstream's per-head mask on the neighbouring line. SAME. Resolve to
  his.
- `top-k.cu` (conflicted in our worktree): his file at b12500 is byte-identical to ours at b12359. Upstream only edited
  lines of the gather kernel that the fork replaces. SAME. Resolve to the fork's file.

## 12. What the 20 Exabit commits collide with on his tree

Measured: `git merge-tree --merge-base=a73c601e6 557a6068f 224b73e24` (the net change of the 20 commits onto his
head). Four files conflict; `mmvq.cu`, `fattn-vec.cuh`, `fattn-tile.cuh`, `ggml/CMakeLists.txt` and
`ggml-hip/CMakeLists.txt` merge by themselves.

| File | Hunks | Cause |
|---|---|---|
| `norm.cu` | 9 | upstream's BF16 templating of `rms_norm_f32` (#32135) under the Q8_1 emission and pre-add of `69e797b1d` / `f83466492`, and his `clamped` plumbing (section 11) where ours had the launch macro |
| `common.cuh` | 1 | `warp_reduce_sum(int)`: upstream's `__reduce_add_sync(GGML_CUDA_FULL_THREAD_MASK, x)` against the DPP branch of `87df3b42e` / `e219d95ec` |
| `gated_delta_net.cu` | 1 | the MMA guard: his `{` (`gated_delta_net.cu:392`) against `if (!keep_rs && !pf.folded())` of `9d70ee129` |
| `ggml-cuda.cu` | 1 | MUL_MAT_ID dispatch (`ggml-cuda.cu:3506-3510`): upstream's `use_mmq_few` against `MMVQ_MMID_MAX_BATCH_SIZE` of `6226b3dba` |

What merges without a conflict and is still wrong or needs a look:

1. `common.cuh`, DPP reductions (`87df3b42e`): three `__shfl_xor_sync(0xffffffff, x, 32, 64)` in
   `ggml_gcn_warp_reduce_sum_f32`, `_max_f32`, `_sum_i32` merge cleanly and then fail HIP's 64-bit-mask assert
   (section 7). They must become `ggml_cuda_shfl_xor(x, 32, 64)`. A compile error, not a silent one.
2. `gated_delta_net.cu`: the resolution has to keep both conditions. `!pf.folded()` is needed on his tree as well (the
   MMA kernel reads q/k/g/beta that a folded call has not computed), and `!keep_rs` is section 3. Neither matters on
   gfx906, both matter on the GPUs of section 3.
3. `ggml-cuda.cu`, MUL_MAT_ID: resolve to `if (!use_mmq_few && ne2 <= MMVQ_MMID_MAX_BATCH_SIZE)`. The sync predicate
   (`ggml_cuda_mul_mat_id_needs_sync`, `ggml-cuda.cu:3449-3484`) merges by itself and still mirrors the dispatch.
   `ggml_cuda_should_use_mmq_few` is a stub that returns false on HIP (`mmq-few.cu:931-943`), so this is text only on
   gfx906.
4. `ggml-cuda.cu`, graph capture: no Exabit commit touches the region where his and our fallback differ (section 9), and
   none uses a name that exists only in ours. The hooks they add (`ggml_cuda_gdn_prefuse_producer` in the node loop,
   `gdn_prefuse_reset` at the start of a graph) land in his loop unchanged. On a re-run after an abandoned chunk his
   code resets the Q8_1 cache like ours (`ggml-cuda.cu:6526`); the GDN prefuse state is not reset on that path in our
   own master either (`224b73e24:ggml/src/ggml-cuda/ggml-cuda.cu:6858`), and the producers stay marked as folded, which
   is what the re-run needs.
5. `norm.cu`: the fused Q8_1 emission reduces with `warp_reduce_max<QK8_1>` / `warp_reduce_sum<QK8_1>` (width 32).
   That stays right on 64-wide wavefronts (each half is one Q8_1 block). Its predicates check F32
   (`ggml_cuda_rms_norm_q8_target`, `ggml_cuda_add_rms_norm_fusable` in `224b73e24`), so upstream's new BF16 RMS_NORM is
   not taken by the fusion; the kernels need to be ported onto upstream's `T` signature.
6. `topk-moe.cu` is not touched by the Exabit commits, but with the DPP reductions in, section 1 does not change: the
   width-32 DPP reduction also stays inside one half.
7. All-reduce size gate (`fb5033219`, `a8d6205f4`), `mmvq.cu`, the CMake commits: no collision with his tree. Upstream
   changed `mmvq.cu` once in between (#32308, an RDNA3.5 gate) and raised the minimum ROCm to 6.3 in
   `ggml-hip/CMakeLists.txt`; both merge by themselves.

## 13. Tests on the gfx906 box, in one place

| Topic | Command or switch | Expected |
|---|---|---|
| 1 softmax widths | his build, `test-backend-ops -o TOPK_MOE -b ROCm0` | softmax cases with 128, 256, 288, 384, 512 experts and no norm fail; pass after the fix |
| 1 softmax widths | a softmax MoE, `GGML_CUDA_DISABLE_FUSION=1` against default | same text / top-5 only after the fix |
| 2 ballot | same run, cases with `n_expert_groups` > 1 | pass without our commit |
| 3 MMA + ring | not possible on gfx906 | - |
| 4 `same_lanes` | `HIP_VISIBLE_DEVICES=0,1 test-llama-archs`, glm5-next "Meta" row | ours about 3.5e-04 (fail), his below 1e-06 |
| 5 prefetch | `GGML_SCHED_DEBUG=2` on each served model | one meta split: prefetch never fires |
| 7 helpers | HIP build of our tree at b12500 | fails at three lines until `9284233fe` is taken |
| 8 pad | `test-backend-ops -o PAD_REFLECT_1D` | 7 of 7 with upstream's file |
| 9 fallback | small `GGML_CUDA_GRAPH_CHUNK_SIZE`, die nearly full | same perplexity as with graphs off |
| 10 bounds | `test-backend-ops -o MUL_MAT_ID`, Q8_0, repack build | pass on both |
