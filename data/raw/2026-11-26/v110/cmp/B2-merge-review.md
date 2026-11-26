# B2: review of the v1.1.0 merge, llama core

Read-only review of the uncommitted merge in `/root/wt-merge-v110` (his `557a6068f` + upstream `v1.1.0`). Line numbers are
in that worktree. Nothing was built or run; every test below is a proposal.

**Result in one paragraph.** I found no place where the ring/replay split yields a wrong result by reading: the three
rollback modes are mutually exclusive, decided once from the same inputs in the context and in the memory, and every
upstream replay line I traced sees the layout it assumes. No fork line of this group was lost in the merge (checked
mechanically, section B.1). What I did find is smaller: the three defects from `B-llama-core.md` are all still there and
one of them gained a second victim from upstream's new code; one new upstream input class bypasses the fork's index
upload (harmless for the one model that uses it); the single-sequence rollback tests no longer reach the replay path for
the Qwen family; and two statements in the merge notes do not match the code (QWEN3NEXT, the qwen4exp clamp). The risk
that reading cannot settle is the replay path through the fork's kernels and meta backend, which is the `-np 8` serving
configuration.

Ordered by how much it matters.

---

## 1. Not a defect I can show, but the first thing to run: replay is now the path of every parallel Qwen context

- `-np 1` keeps the ring; `-np N>1` with `n_rs_seq > 0` on qwen35 / qwen35moe now replays
  (`src/llama-context.cpp:369-386`). That is the recommended serving configuration (`-np 8`), and before the merge it
  ran the plane-indexed snapshots.
- What reading cannot answer: the replay graph (`src/models/delta-net-base.cpp:159-306`) through the fork's meta
  backend and its split rules, which came from upstream's meta (`src/llama-model.cpp:820-827`, `:932-940`, `:997-1014`,
  `:1044-1090`, `:1145-1201`). Upstream's own arch test still skips rollback under tensor split with "the meta backend
  aborts with n_rs_seq > 0" (`tests/test-llama-archs.cpp:1607`), for qwen4exp and glm5-next.
- Known to you already: a replay op carries its replay inputs in every ubatch, also with nothing pending, so prefill
  in a replay context stays off the chunked and column-thread kernels.
- Test: Qwen3.8-27B Q8_0 + MTP, `-sm tensor`, temperature 0, same requests under `-np 1` (log: `snapshot ring enabled`,
  `rs_replay = false`), `-np 4` (log: `rs_replay = true`, memory line ends `rs_seq replay`) and
  `LLAMA_RS_REPLAY=0 -np 4` (plane-indexed). Compare text, draft acceptance, prefill and decode rate. Then a draft
  longer than one rejection per request, so partial rollbacks happen.

## 2. The three defects of his tree: all still present (question C)

Upstream did not touch any of the three sites between b12500 and v1.1.0.

- **Carried MTP row at position 0.** Unchanged: upload at `common/speculative.cpp:2619-2625`, reset after it at
  `:2634-2637`. Same scenario and test as `B-llama-core.md` 1.2.
- **KV-only MTP store without the FP8 scale.** Unchanged: `src/llama-graph.cpp:3775-3800`; `build_attn` still divides
  (`:3722`, helper at `:3659`). Still latent (needs a GGUF with KV scale tensors).
- **LiliCorr on the deferred path.** Unchanged (`common/speculative.cpp:1218`; the replay `:1296-1398` never calls
  `store_context_features`), and see item 3.

## 3. The deferred DFlash replay now also starves upstream's new DSpark Markov head (latent, same class as LiliCorr)

- Upstream added a second consumer of `store_context_features`: a DSpark drafter with `dflash.has_markov_context`
  (Inkling2 DSpark, `common/speculative.cpp:1767`, override at `:1804-1823`). It reads the context state of position
  `pos0 - 1` and does not draft without it (`:1867-1870`). Upstream now calls the hook on the device path too (`:1575`)
  as well as the host path (`:1640`).
- The fork's deferred replay stores nothing, so whenever the deferral is live (host copies: `LLAMA_DFLASH_DEV=0`, or the
  device buffer not usable) the first draft after a prompt has no state. Unlike LiliCorr this one checks the stored
  position (`:1825-1831`), so the result is "no draft", not a wrong one. It is masked the same way: the server's 4-token
  tail batch goes through the eager path when prompt checkpoints are on.
- On his default (device buffer) it works, because that path is eager.
- Fix that covers both heads: call `store_context_features(pc.seq_id, pc.n)` after each decode in
  `flush_pending_upto`.
- Test: none possible here, no Inkling2 DSpark model on the box. LiliCorr: as in `B-llama-core.md` 5.2.

## 4. A. Ring / replay coexistence

### A.1 Places that merged without a conflict (`src/llama-memory-recurrent.cpp`)

No ring path is reachable with `rs_replay`, and no replay path with the ring. What I checked, one by one:

- **Decision.** Context: `rs_replay` is cleared when `ring_supported(model, n_seq_max, n_rs_seq, offload_kqv)` or
  `LLAMA_RS_REPLAY=0` (`src/llama-context.cpp:375-386`). Memory: `ring_enabled = !rs_replay && ring_supported(...)`
  with the same four values (`:93-94`; they reach it unchanged through `create_memory`, `src/llama-context.cpp:594`).
  Nothing changes `n_seq_max`, `n_rs_seq` or `offload_kqv` after line 369.
- **Rows per cell.** Replay allocates one row per cell, the other two modes `1 + n_rs_seq` (`:175`). Every fork line
  that indexes a plane resolves to plane 0 under replay, because `rs_idx` is only ever set in the non-replay branch of
  `seq_rm` (`:336-343`): `s_copy()` (`:1745-1763`), `state_write` (`:1082-1108`). The only loose end is the bound in the
  fork's assert, which still multiplies by `n_planes()` (`src/llama-graph.cpp:615`): weaker than it could be, not wrong.
- **find_slot.** The ring check is guarded (`:766-778`). The replay block is guarded (`:945-949`). `rs_max` is shared and
  only read outside the ring (`:332-334`).
- **Cell swap.** Upstream's new `src_cell.rs_max = 0` outside replay (`:910-916`) is right for the plane-indexed
  fallback: the moved-out cell only gets its final state copied. It cannot affect the ring, which has one cell.
  It also fixes a bug his b12500 tree had in parallel contexts (a moved sequence inherited the cell's depth).
- **seq_cp, seq_keep, clear.** `reset_ring()` runs in all modes and only touches ring vectors and `rs_idx`
  (`:245-246`, `:284-287`, `:438-442`). A freed cell keeps a stale `rs_pend`, which upstream neutralises through
  `cell.src < 0` (`:947`).
- **Snapshot planner.** Reached only through `fill_s_write`, which returns unless `uses_ring()`
  (`src/llama-graph.cpp:552`).
- **Graph inputs.** Ring inputs exist iff `uses_ring()` (`src/llama-graph.cpp:4470-4485`), replay inputs iff the
  memory replays (`:4504-4534`). The builders branch on `cparams.rs_replay` first, then on `inp->s_write`
  (`src/models/delta-net-base.cpp:62`, `:78`, `:318`, `:377`; `src/models/qwen4exp.cpp:1434`).
- **`set_input_rep()`** (`src/llama-graph.cpp:643-686`): the hand-written version matches upstream's values, and the
  separate narrow window gets the first `n` columns of each sequence, which is what upstream's view was. The guards on
  unallocated inputs are an improvement: upstream dereferences `rep_cols->data` unconditionally.
- **`can_reuse`** (`:688-722` and the hybrid variants `:1424-1570`): nothing is needed for replay, the replay graph is a
  function of the ubatch shape that `allow_reuse` already compares.

Two things that are not bugs today:

- `llm_graph_input_mem_hybrid_iswa_rel::set_input` (`src/llama-graph.cpp:1604-1616`) is new upstream code that merged
  verbatim. It writes `s_copy->data` through the host pointer and never fills the fork's separate `s_copy_main` /
  `s_copy_tail` inputs, and never calls `fill_s_write`. Its only user is Inkling2 (`src/models/inkling2.cpp:268`,
  `:307`), which reads state with the default getter and is not a ring architecture, so only `s_copy` is in the graph.
  It breaks the day that input is used with a custom state getter. One line makes it consistent:
  `inp_rs->upload_s_copy(mctx->get_recr(), ubatch->n_seqs)`.
- The constructor's "memory module is empty" branch assigns the parameters, not the members (`:220-226`), so
  `ring_enabled` and `rs_replay` stay set on an empty memory. Upstream has the same shadowing. No graph reads them then.

### A.2 The plane-indexed fallback still works with v1.1.0's memory code, by reading

- Memory: rows `1 + n_rs_seq` (`:175`), bound `cell.rs_max` (`:334`, set at `:944`), plane chosen by `rs_idx`
  (`:1757-1759`). Graph: his code unchanged (`src/models/delta-net-base.cpp:331-416`; the only line the merge removed
  from his file is the `K = 1` call in `build_delta_net`), conv through upstream's `build_conv_state_store`, which kept
  its snapshot slots (`src/llama-graph.cpp:4593-4622`). `ggml_gated_delta_net(..., K)` still takes `K > 1`
  (`ggml/src/ggml.c:6476-6485`).
- **QWEN3NEXT never reaches it**, nor the ring: it is not in `llm_arch_supports_rs_rollback`
  (`src/llama-arch.cpp:1558-1582`), so `n_rs_seq` is clamped to 0 at `src/llama-context.cpp:193-196`, in his tree and
  upstream alike. Its entry in `ring_supported()` (`:50`) is dead.
- **Parallel qwen4exp never reaches it either**: `n_rs_seq` is clamped to 0 for `n_seq_max > 1`
  (`src/llama-context.cpp:197-203`). That clamp was written against "the unproven plane-indexed path"; with replay
  available it now only takes rollback away from parallel qwen4exp. A decision, not a bug.
- `LLAMA_RS_REPLAY=0` also moves Clover-MoE, the KDA models and RAON-1 to plane-indexed snapshots. Their non-replay
  branches exist (`build_kda_conv1d`, `src/models/delta-net-base.cpp:129-139`), but RAON-1 is new in v1.1.0 and nobody
  has run it that way.
- Test: `LLAMA_RS_REPLAY=0`, qwen35 fixture or model, `test-llama-context` case `replay` (two sequences,
  `tests/llama-context/cases-rs.cpp:24-120`), on HIP.

### A.3 `rs_replay` reaches the memories consistently

- `llama_memory_hybrid` and `llama_memory_hybrid_idx` get `cparams.rs_replay` (`src/llama-model.cpp:3811`, `:4150`,
  `:4171`). A pure recurrent memory and `llama_memory_hybrid_iswa` are hard-wired to `false` (`:4026`,
  `src/llama-memory-hybrid-iswa.cpp:61-62`), as upstream.
- That is consistent only because no replay architecture is pure-recurrent or SWA-hybrid today. If one ever is, the
  context says replay, the memory has no pending tensors, and `build_recurrent_attn_replay` dereferences a null
  `get_spk_l()` (`src/models/delta-net-base.cpp:251-257`). Upstream has the same hole.

### A.4 State save / load across modes: fails cleanly, cannot load

- The stream has no mode marker and `LLAMA_SESSION_VERSION` / `LLAMA_STATE_SEQ_VERSION` did not change
  (`include/llama.h:46`, `:49`). A replay stream has `rs_pend` per cell (`:1199-1201`), wider conv rows, and the pending
  tensors at the end (`:1300-1314`); a ring or plane stream has none of them.
- Ring/plane stream into a replay context: the meta reader takes the next 4 bytes as `rs_pend` (`:1349-1351`), the data
  reader then reads the layer count from the wrong word and stops at "mismatched layer count" (`:1439-1442`). Replay
  stream into a ring/plane context: the unread `rs_pend` shifts the data header the same way. An empty state fails at
  the row-size check (`:1470-1473`). For a single-sequence restore I found no alignment that passes: the word read as the
  layer count is 0 in both directions. A whole-context file with several cells loses alignment already in the meta loop
  (`:1385-1418`); I expect it to end in one of the range checks there or in the same layer-count check, but I did not
  enumerate it.
- After the failure the hybrid memory clears the attention half too (`src/llama-memory-hybrid.cpp:198-206`), the context
  returns 0 (`src/llama-context.cpp:4534-4537`), and the server re-processes the prompt.
- When it happens: slot files or session files written under `-np 1` and read under `-np N`, or the reverse; files from
  before the merge written by a parallel context; a run with `LLAMA_RS_REPLAY` toggled. Never inside one process.
- Test: `--slot-save-path`, save under `-np 1`, restart with `-np 4`, restore: expect the error line and a full
  re-process, no abort.

## 5. Tests no longer cover what their names say (`tests/llama-context/cases-rs.cpp`)

- `rollback` and `rollback_bound` create one-sequence contexts (`:206-207`, `:410-411`), so for the Qwen family they
  exercise the ring, as in his tree. Upstream wrote them for replay. Replay for qwen35 is covered only by the
  two-sequence case `replay` (`:24-120`), and for qwen4exp by nothing (clamped, the case skips).
- There is no switch to force replay on a context the ring covers; `LLAMA_RS_REPLAY` only turns it off. A value that
  forces it (for example `LLAMA_RS_REPLAY=2`) would put upstream's single-sequence cases on the replay path of the
  fork's kernels.
- New from upstream and worth running on HIP: the runner now also fills the recurrent state with NaN before each case
  (`tests/llama-context/test-runner.cpp:77`, `test-util.cpp:302-340`). On the ring that is a direct test that no
  rollback reads a plane it never wrote.

## 6. B. Silent breaks in the other files of the group

### B.1 Mechanical check: no fork line lost

For every file of the group I took the lines his tree adds over upstream b12500 and looked for each in the merged file.
Missing: 2 lines in `src/llama-graph.cpp` (the two `build_get_k/v` calls that became `gctx.` calls), 12 in
`src/llama-context.cpp` (the `LLAMA_NO_FUSED_LID` block), 4 in `src/llama-memory-recurrent.cpp` (the ring condition, now
`ring_supported()`). All three are in your merge notes. Zero in the other 30 files checked, among them
`common/speculative.cpp` (514 fork lines), `tools/server/server-context.cpp`, `src/llama-kv-cache.cpp`,
`src/llama-model.cpp`, `src/llama-vocab.cpp`, `src/models/dflash.cpp`, `src/models/models.h`, both test files.

### B.2 `src/llama-graph.cpp`: no bypass of the replica helpers left

- The merged file has no direct `mctx_cur->cpy_k / cpy_v / get_k / get_v` outside the helper bodies (`:3306`, `:3319`,
  `:3339`, `:3351`); upstream v1.1.0 has 14, all routed. The four in `build_attn_iswa_impl` that you fixed were the only
  ones the refactor exposed.
- Helper use in the model files is the same as in his tree (only `src/models/deepseek4.cpp`, 2 calls). The direct calls
  in the model files (dots3note, kimi-k31, hy-v4-flash, glm5-next, qwen4exp, minimax-m3, ...) were direct in his tree
  too.
- `build_rs`: upstream changed the zero state from `scale_inplace(0)` to `fill_inplace(0.0f)` (`:4429`). The fork's meta
  backend and CUDA both handle `GGML_OP_FILL` (`ggml/src/ggml-backend-meta.cpp:1512`), so I expect no problem, but it is
  a new op on a view of a split cache tensor: covered by the run in item 1.
- `llm_graph_params` lost `hparams`, `prec_policy`, `hdmd` upstream. The positional initialiser in
  `llama_context::graph_params` (`src/llama-context.cpp:3680-3706`) matches the struct order
  (`src/llama-graph.h:870-921`), the fork's `layer_tap_dev` last.

### B.3 Weight scales (#32669): no stale argument found

- `build_lora_mm_id` lost its `w_s` parameter and `build_moe_ffn` three scale parameters, all `ggml_tensor *`, so a
  stale argument would have bound to `slots` / `selected_experts_in` without a compile error. No call passes one: the
  remaining `build_lora_mm_id` calls are `src/llama-graph.cpp:2731`, `:2746`, `:2755`, `:2865` and
  `src/models/k2-horizon.cpp:184`, and no `_exps_s` is used as an argument anywhere in `src/models` or
  `src/llama-graph.cpp`.
- `build_lora_mm` still takes an explicit scale and falls back to the registry (`:2007-2028`), so the fork's explicit
  `wq_s` / `wk_s` / `wv_s` calls apply the scale once. I found no manual `ggml_mul(..., *_s)` after a `build_lora_mm`
  in fork code.

### B.4 `src/llama-kv-cache.cpp`: UNSURE on one error path

`state_clear` moved from `llama_clear_tensor_data` (always `set_tensor`) to `ggml_backend_tensor_memset` (`:3188-3228`,
and `src/llama-memory-recurrent.cpp:1610-1620`). Under `-sm tensor` that now lands in the fork's own
`ggml_backend_meta_buffer_memset_tensor`, which asserts `ne[2] == 1` and row-aligned ranges for split tensors
(`ggml/src/ggml-backend-meta.cpp:1906-1940`). A K cache with more than one stream is 3D. This only runs after a failed
state restore. I did not check whether the old `set_tensor` path accepted those shapes, so I cannot say it got worse.
Test: the cross-mode restore of A.4 under `-sm tensor -np 4` with a non-unified KV.

### B.5 Looked at, nothing found

- `common/speculative.cpp`: the merge only brings upstream's DSpark Markov head; the MTP class is untouched;
  `draft_prepare()` is still called by all four heads (`:1650`, `:1846`, `:2035`, `:2219`).
- `tools/server/server-context.cpp`: upstream's changes (decision groups, idle-slot caching with a retry, slot
  selection by prompt cache) do not touch the fork's lines.
- `src/llama-context.cpp`: `graph_max_nodes()` now asks the model (`:3530`); the DFlash2 rule moved to
  `src/models/dflash.cpp:142-153`; no fork row was in the old chain.
- `src/llama-model.cpp`: `lid_k_hdmd()` replaced an architecture list; his V4.1 keeps the old behaviour
  (`src/models/models.h:1562`).
- `src/llama-vocab.cpp`: the no_vocab handling is intact and `print_info` has no unguarded `id_to_token.at()` left.
- `src/models/dflash.cpp`, `src/models/models.h`, `tests/test-llama-archs.cpp`: fork lines intact; the rest is the
  DeepSeek reader's.

## 7. Corrections to the merge notes

- "plane-indexed fallback ... and QWEN3NEXT": QWEN3NEXT has `n_rs_seq == 0` always (A.2).
- "every other context of those models with n_rs_seq > 0 replays": true, but parallel qwen4exp has `n_rs_seq == 0`
  by the fork's clamp, so it neither replays nor rolls back (A.2).
- Worth one line in the message: a state saved by a ring context does not load into a replay context and the reverse
  (A.4).
