# B: llama core (context, graph, KV cache, recurrent memory, speculative, server)

Read-only comparison. "His" = mxxm-t at `557a6068f` (line numbers are in his tree, worktree `/root/wt-mxxm-b12500`,
unless marked "ours"). "Ours" = `a73c601e6` carried to b12500 in `/root/wt-cmp-ours-b12500` (uncommitted, three of my
files with conflict markers). Nothing was built or run; every "test" below is a proposal. Existing measurements are cited
where they answer the question.

Verdict counts: OURS-FIXES-HIS 4 (one reachable on the serving configuration, one resource-only, two latent),
HIS-BETTER 4, UNSURE 2, UPSTREAM-MOVED 3, EQUIVALENT 8, SAME 9. Not mine (DeepSeek reader): 5 hunks, listed in section 8.

## Short answers to the five questions

1. **MTP rows.** He sequences them the same way we do: deferred prefill first, device rows only after the replay has
   released the accum buffer (`common/speculative.cpp:2381-2384`, `2678-2681`). His deferred prefill cannot replay rows
   that were never extracted through the device-row interaction, because the device rows are off for the whole window in
   which the accum buffer exists. He has no context-level guard (ours refuses each mode while the other is on) and no
   `LLAMA_MTP_DEV` switch. `5ab83d630` brings his adaptive disable to parity with our `dev_rows_disable()`. One defect in
   his: the fork's "zero the carried row at position 0" runs after the device upload, so it does nothing on device rows
   (section 1).
2. **KV-only MTP store.** `977ff5061` = ours for the per-layer rotation and `build_rope`. He did not add the FP8 scale
   division that came with #28392 at b12500 (section 2, latent).
3. **Snapshot ring.** The rollback bound is the same code in both trees, so neither keeps more on CPU/CUDA/HIP/meta. His
   `1c8f7fa09` adds a backend gate (ring off on any other backend), a once-per-ubatch assert and a runtime query. His
   tests check that an accepted deep rollback restores the right state but do not require the ring to accept it; ours
   require the exact bound but never check the restored state (section 3).
4. **llama-context / llama-graph.** No difference in state save/load, sequence copy or graph reuse. Naming only for the
   taps (`layer_tap_dev` his, `layer_inp_tap` ours). He still allocates the fork's static tap tensors when upstream's
   device buffer is on, and keeps one dead helper (section 4).
5. **Server and drafters.** `tools/server/server-context.cpp` is identical apart from one condition's operand order. In
   `common/speculative.cpp` he kept upstream's order for DFlash (device buffer first, deferred accum armed afterwards and
   then never filled), and LiliCorr is not kept off the deferred path (section 5).

---

## 1. MTP hidden rows: host, device, deferred (`common/speculative.cpp`, `src/llama-context.cpp`)

### 1.1 Sequencing of deferred prefill and device rows: EQUIVALENT

- His: the constructor arms the accum (`:2362-2376`) and calls `set_dev_rows()` only `if (!deferred_prefill)`
  (`:2381-2384`); `finish_deferred_prefill()` frees the accum (`:2678`) and then calls `set_dev_rows()` (`:2681`).
  `process_decode()` uses `dev = dev_rows && h_tgt_in == nullptr` (`:2454`), so a replay always takes host rows.
- Ours: `dev_rows_enable()` returns while `prefilling` (ours `:2285-2300`), is called from the constructor, from
  `finish_deferred_prefill()` and from `begin()`; `process_decode()` asserts `h_tgt_in == nullptr` under `dev_rows`.
- Why equivalent: in both trees `embeddings_nextn_dev` is false on the target for as long as `embd_pre_norm_accum` is
  set, so the host extraction at `src/llama-context.cpp:2902` (`!write_h_nextn`) runs and fills the accum
  (`:2913-2953`). Deferred prefill is armed once per process in both (`n_seq == 1`, `:2357`), and `-np > 1` gets device
  rows from the constructor in both.
- Ours only, not a behaviour difference under the server: `llama_set_embeddings_nextn_dev` refuses while the accum is
  set and `llama_set_embeddings_pre_norm_accum` refuses while device rows are on (ours `src/llama-context.cpp:1753`,
  `:1835`). His context has neither check (`:1748-1763`, `:1825-1864`); the invariant is held by the caller's order only.
  `LLAMA_MTP_DEV=0` (host rows after the replay) exists only in ours; in his tree there is no way to A/B device rows
  against host rows without a rebuild.
- Existing measurement (ours, `reports/2026-11-14-v100-rebase.md` section 2, `-np 1`, n = 1): deferred then device 548.6
  tok/s prefill, host rows 556.6, device from the start 539.0; decode and acceptance (154/200) the same in all three.

### 1.2 The carried row of a sequence that starts at position 0: OURS-FIXES-HIS (reachable on gfx906)

- His: the pending rows are uploaded to the device buffer first (`:2487-2494`), and the fork's reset
  (`017a5d3f8` "zero MTP carrier at sequence start") sits inside the token loop after it (`:2503-2506`). On the device
  path the first token of the sequence reads device row `n_batch_tgt + seq_id` (`:2513-2515`), which already holds the
  stale `pending_h`; the zeroed host copy is never used and is overwritten at `:2580`.
- Ours: the reset runs for every sequence before the upload (ours `:2561-2576`).
- Scenario: Qwen3.8-27B with its MTP head, `-np 8` (device rows from the first request, `:2357`, `:2382`) or `-np 1` from
  the second request on. Any request that is processed from position 0 (no reusable prefix, `cache_prompt: false`, a
  hybrid slot forced to full re-processing) pairs its first token with the last hidden row of the slot's previous
  request instead of zeros. The first request of a slot is correct (the row is still the constructor's zeros), so two
  identical requests build different head KV at position 0.
- Consequence: one wrong row in the head's prompt KV. Drafts are verified by the target, so the text is unchanged under
  greedy verification; what can change is draft acceptance and the run-to-run reproducibility the fork commit was
  written for. I have not measured the size of the effect and expect it to be small.
- Test: his build, Qwen3.8-27B Q8_0 + MTP, `-sm tensor`, `-np 1` with `LLAMA_MTP_DEFERRED_PREFILL=0` (device rows from
  the start), temperature 0, `cache_prompt: false`, requests A, B, A. Compare `timings.draft_n_accepted / draft_n` of
  the two A requests; on our build they must be equal and `LLAMA_MTP_DEV=0` gives the host-row reference. Equal counts on
  his build would mean the defect is benign for that prompt, not absent.
- Fix for a PR: move the reset loop ahead of the upload, 5 lines.

### 1.3 `5ab83d630` (reset verify rows when extraction is disabled): EQUIVALENT

- His adaptive disable turns the extraction off through `llama_set_embeddings_nextn(ctx_tgt, false, false)`, which also
  drops the device buffer inside the context (`src/llama-context.cpp:1564-1568`); his `dev_rows` flag stays true and
  `begin()` re-allocates through `if (dev_rows) set_dev_rows()` (`:2419-2422`). Before `5ab83d630` the per-sequence
  `verify_row0` / `verify_h_rows` survived that, so an `accept()` for a sequence that was not in the first batch after
  the re-enable read a row of the freshly allocated, uninitialised device buffer into `pending_h` (`:2919-2923`). The
  commit clears both arrays in `disable_seq()` (`:2243-2244`). It also moves a trace line and adds a comment in
  `ggml/src/ggml-backend.cpp` saying upstream's input prefetch is unused in his tree (ggml reader's topic: ours folded
  the prefetch into the fork's copy loop).
- Ours had this from the start: `dev_rows_disable()` zeroes `verify_h_rows` (ours `:2302-2311`), so `accept()` returns
  at the `n_rows <= 0` check.

### 1.4 `accept()` guard: HIS-BETTER (defensive only)

- His `accept()` reads device rows only `if (dev_rows && verify_row0[seq_id] >= 0)` (`:2919`) and the host path sets
  `verify_row0 = -1` (`:2584`). Ours tests `dev_rows` alone (ours `:2997`). After our deferred replay the verify rows
  are host rows with `verify_row0 == -1`; an `accept()` before the next `process()` would read device row
  `-1 + n_accepted`. I found no such order in the server (every decode is followed by `common_speculative_process`,
  `tools/server/server-context.cpp:4777`, before any `accept`, `:5019`), so this is robustness, not a reachable bug.

### 1.5 Draft context after a disable / re-enable cycle: UNSURE

- Ours clears and re-sets `ctx_state_src`, which flags a re-reserve of the draft context each time
  (`src/llama-context.cpp:1790-1796`). His leaves `ctx_state_src` pointing at the target across the cycle, so the draft
  context relies on `allow_reuse` comparing the buffer tensor pointer (`src/llama-graph.h:931-932`) to notice the new
  device buffer. I would need a run that forces the adaptive disable (a non-finite draft probability; there is no
  switch for it) to say whether his path ever reuses a graph bound to the freed buffer. Untested in both trees.

## 2. KV-only MTP store (`src/llama-graph.cpp`, `src/models/qwen35.cpp`)

### 2.1 Per-layer rotation and `build_rope` (`977ff5061`): SAME

`inp->self_k_rot[il]` / `self_v_rot[il]` with `llama_mul_mat_hadamard` (`src/llama-graph.cpp:3623-3628`) and
`Kc = build_rope(Kc, inp_pos, il)` in the replay graph (`src/models/qwen35.cpp`, identical file in both trees).

### 2.2 FP8 cache scales: OURS-FIXES-HIS (latent, not reachable with the GGUFs on this box)

- His `build_attn_store_kv` (`:3615-3640`) still says it "mirrors the K/V-store block of build_attn exactly"
  (`:3620-3622`), but since b12500 `build_attn` stores `K/s` and `V/s` and multiplies `Q` and the output by `s`
  (`build_attn_scale_kv_inputs`, `:3488-3510`, called at `:3553`; output at `:3588`). The KV-only store writes unscaled
  K and V. `977ff5061` predates #28392, so this is a silent break of the b12500 merge.
- Ours divides by `get_k_scale(il)` / `get_v_scale(il)` after the rotation (ours `src/llama-graph.cpp:3611-3619`).
- Scenario: `-np 1`, `--spec-draft-type-k f8_e4m3 --spec-draft-type-v f8_e4m3`, and a GGUF that carries KV scale tensors
  for the MTP block (`blk.N.attn_k.k_scale`, loaded at `src/llama-model.cpp:2304-2305`, written by
  `conversion/base.py:347` from checkpoints that ship `.k_scale` / `.v_scale`). The first request's replay then stores
  prompt rows `s` times too large in both K and V, and the head attends to them with `Q*s`; later rows are stored
  correctly. Result: poor drafts on the first request, text unchanged. qwen35 is not in the FP8 exclusion list
  (`src/llama-context.cpp:5131-5143`), so the context is created.
- Reachability: without scale tensors `get_k_scale` returns null (`src/llama-kv-cache.cpp:1594-1602`) and both trees
  behave the same. `/root/models/Qwen3.8-27B-Q8_0.gguf` has none (header scan: 0 occurrences of `k_scale` / `v_scale`),
  so nothing served here hits it today.
- Test: needs a scaled GGUF; then `LLAMA_MTP_NO_KV_ONLY=1` against the default must give the same acceptance. A synthetic
  fixture in the style of `tests/test-fp8.cpp` is the realistic way.

## 3. Snapshot ring against upstream's rollback bound (`src/llama-memory-recurrent.*`, `tests/llama-context/cases-rs.cpp`)

### 3.1 The bound itself: SAME

`max_rollback = ring_enabled ? min(n_rs_seq, valid_depth - 1) : cell.rs_max` in both
(`src/llama-memory-recurrent.cpp:286-288`; `rs_max` from `:885`). Neither tree keeps more than the other where the ring
is on.

### 3.2 Backend gate: HIS-BETTER (not on gfx906)

- His `1c8f7fa09` turns the ring off unless every device of the model is CPU, CUDA, ROCm or Meta
  (`:21-45`, used at `:74-77`), with the reason in the comment: the ring needs every snapshot slot written by the
  delta-net kernels, verified on those backends only. Ours enables the ring on any backend.
- Effect on this box: none (`GGML_CUDA_NAME` is "ROCm", the tensor-parallel device is "Meta"). On Vulkan, Metal, SYCL
  his tree falls back to upstream's last-ubatch bound; ours would accept deeper rollbacks there. Whether those backends
  actually leave slots unwritten I did not check, so "ours is wrong there" is not claimed, only that his cannot be.

### 3.3 Once-per-ubatch assert: HIS-BETTER (minor)

`plan_snapshot_writes` asserts it runs once per ubatch (`:608-610`, member at `src/llama-memory-recurrent.h:233`); a
second plan would advance the ring twice. The only caller is `llm_graph_input_rs::fill_s_write`
(`src/llama-graph.cpp:517-535`), reached once per `process_ubatch`, so I see no spurious trigger.

### 3.4 Tests: HIS-BETTER, with one thing of ours worth adding

- His: asks the context (`llama_memory_rs_ring_active`, `src/llama-context.cpp:5261`, `src/llama-ext.h:39`). With the
  ring on, every rollback deeper than upstream's bound that is accepted must replay like a fresh context (NMSE <= 1e-4,
  `cases-rs.cpp:286-331` and `:441-481`); a refused one is skipped (`:316-318`, `:456-458`). Upstream's guaranteed
  rollback must still be accepted.
- Ours: predicts the ring from an architecture list and requires acceptance exactly up to
  `min(n_rs_seq, n_prefix + n - 1)` and refusal beyond (ours `cases-rs.cpp:209`, `:296`, `:399`), with no check of the
  state a deep rollback restores.
- So his tests cover the failure that matters (a wrong state restored silently) and follow the code on any backend, but
  a ring that refused everything beyond upstream's bound would pass them. Ours pins the bound but would pass a ring that
  restores garbage, and the static list is wrong wherever his backend gate switches the ring off. The union is the
  right test: his structure plus a `CTX_REQUIRE` that the rollback is accepted within the ring bound.
- Test: `test-llama-context` (group rs) on HIP, both trees.

## 4. `src/llama-context.cpp`, `src/llama-graph.cpp`, `src/llama-kv-cache.cpp`

- **State save/load, sequence copy, graph reuse: SAME.** The diff of his merge against our carry has no hunk in those
  paths; the hunks are the ones listed here.
- **Tap naming: SAME.** Both keep upstream's `layer_inp_dev` for upstream's buffer. The fork's persistent taps are
  `layer_tap_dev` in his tree (`src/llama-context.h:442-444`, `src/llama-graph.h:858`) and `layer_inp_tap` in ours. Any
  patch of ours that touches them has to use his name.
- **Static taps under the device buffer: OURS-FIXES-HIS (resource only).** His `sched_reserve` allocates the persistent
  tap tensors whenever taps exist (`src/llama-context.cpp:943`), also when `embeddings_layer_inp_dev` is on and
  `set_outputs` skips them (`src/llama-graph.cpp:1602`). Ours adds `&& !cparams.embeddings_layer_inp_dev` (ours `:940`).
  Cost in his tree: taps x n_embd x n_ubatch x 4 bytes on the output device, unused (DSpark on DeepSeek-V4.1-Flash: 3 x
  5120 x 512 x 4 = 31 MB at `-ub 512`). Counted together with the inert accum in 5.1.
- **KV shift with replicas: EQUIVALENT.** His looks the owner's `k_rot` up inside the loop
  (`src/llama-kv-cache.cpp:2351-2366`), ours stores it in the list; same tensor in both.
- **Dead helper: SAME (cleanup).** His keeps an unused `static ggml_mul_mat_aux` (`src/llama-graph.cpp:135`), which
  upstream replaced with `llama_mul_mat_hadamard`. Unused-function warning only.
- **Mask left unallocated for KV-only graphs: SAME** (comment wording differs).

## 5. Drafters in `common/speculative.cpp` and the server

### 5.1 DFlash / DSpark feature source: EQUIVALENT in output, different default; one OURS-FIXES-HIS (resource) and one UNSURE

- His (upstream's order): device features are decided first (`:1186-1201`), then the deferred accum is armed regardless
  (`:1216-1224`). With the device buffer on, the target never writes the accum (`src/llama-context.cpp:3234-3236`), so
  `llama_layer_inp_accum_span_epoch` returns 0 (`:1948-1965`), `process()` falls through its deferral
  (`common/speculative.cpp:1498-1500`) and takes the device path (`:1559-1576`). The fork's deferred prompt encode is
  therefore dead by default wherever the device buffer is usable, while the log still prints "deferred prompt encode
  enabled". `LLAMA_DFLASH_DEV=0` brings it back.
- Ours: deferred encode decided first, device features only when it is not armed (`LLAMA_DFLASH_DEFER=0`).
- Output: measured on this box on our build, DSpark on DeepSeek-V4.1, `-sm layer`, `-c 8192 -np 1`, n = 1
  (`data/raw/2026-11-14/drafters.progress.txt`): same text and acceptance on all three paths; 20.1 / 25.3 tok/s
  deferred, 21.1 / 25.9 device buffer, 20.5 / 25.9 host copies. The stock run on the same setup logged "DFlash target
  features via device buffer", so his tree takes the device path there.
- **OURS-FIXES-HIS (resource, reachable with DSpark):** on that path his tree allocates and zero-fills the accum and
  never uses it: taps x n_seq x n_ctx_seq x n_embd x 4 bytes of host memory, pinned where the output device has a host
  buffer type (`src/llama-context.cpp:1895-1922`). DSpark on DeepSeek-V4.1-Flash (3 taps, n_embd 5120): 0.5 GB at
  `-c 8192`, 8 GB at 128K. Plus the static taps of section 4. No wrong result.
- **UNSURE:** which default is faster on long prompts. The deferral was written for prefill time and the only numbers
  are 192-token generations after short prompts. Test: his build, DSpark, a prompt of 16K tokens or more, time to first
  token with the default against `LLAMA_DFLASH_DEV=0`.
- Ours also times the device-path inject for the stats (ours `:1579`); his device path is untimed (`:1569`).

### 5.2 LiliCorr on the deferred path: OURS-FIXES-HIS (latent, masked by the server's default)

- His deferral gate does not exclude LiliCorr (`:1217-1218`); LiliCorr has `dev_features = false` (`:2058`), so it runs
  on host copies with the accum live and its prompt chunks of 16 rows or more are deferred (`:1482-1501`). The replay
  (`flush_pending_upto`, `:1296-1398`) never calls `store_context_features` (only the eager path does, `:1639`), and
  `draft()` reads the slot of position `pos0 - 1` without checking `lilicorr_pos` (`:2112-2114`). A first draft whose
  anchor row came through the replay reads zeros, or a row left by an earlier request.
- Ours: `want = ... && !is_lilicorr` (a73c601e6 `common/speculative.cpp:1220`, listed in `RELEASE-RUNBOOK.md` section 3
  as a silent break); our 2026-11-14 LiliCorr log (`/root/rebase-v100/drafters/nm-pr.log`) shows "deferred prompt encode
  disabled".
- Why it is masked: when the server makes prompt checkpoints it decodes the last 4 prompt tokens as a batch of their own
  (`tools/server/server-context.cpp:4480-4498`, gated by `do_checkpoint`, `:4359-4372`). Four rows are below the
  deferral threshold, go through the eager path and store exactly the rows the first draft needs. That gate is on for
  targets with bounded or no partial removal, which I believe includes Nemotron-H, but I did not confirm it from a log.
- Scenario where it shows: Nemotron 3.5 + LiLiCorr with `--ctx-checkpoints 0` (or any caller of `common_speculative`
  that feeds the prompt as one batch): the first draft block of each request is built on unset features.
- Test: his build, Nemotron + LiLiCorr, `--ctx-checkpoints 0`, many short generations (n_predict near n_max), acceptance
  with the default against `LLAMA_DFLASH_DEFER=0`.
- Fix for a PR: a `dflash_options` flag that keeps LiliCorr eager, or `store_context_features(pc.seq_id, pc.n)` after
  each decode in `flush_pending_upto`.

### 5.3 DFlash heads split into subclasses: UPSTREAM-MOVED (his adaptation is the one to keep)

Upstream split `draft()` into `dflash`, `dspark`, `dflash2` and `lilicorr` classes between b12359 and b12500. His
`draft_prepare()` (`:1233-1247`) gives all four the deferred flush and the null-anchor guard (`:1649`, `:1778`, `:1903`,
`:2087`). Our carry has them only in the base class (ours worktree `:1662`, `:1685`): a mechanical rebase of ours would
drop the null-anchor guard for DSpark, the drafter tested here. The same split explains the two other conflict hunks in
our worktree (the LiliCorr / DFlash2 constructor block and `opt.dev_features`): UPSTREAM-MOVED.

### 5.4 `tools/server/server-context.cpp`: SAME

Identical except the operand order of one condition (`:686`). Both members of the one conflict in our worktree
(`prompt_checkpoint_storage_lossless`, `n_swa_reuse`) are present in his (`:1128`, `:1131`); the local `spec_i_batch`
with per-row probabilities is the same (`:4949`, `:5062-5069`). MTP gating by `n_seq`, per-sequence DFlash accum and
the rollback-versus-checkpoint rule (`:4981-4983`) do not differ.

## 6. `src/llama-model.cpp` (non-DeepSeek)

- **Hybrid memory filters: EQUIVALENT; ZONOS3 is UPSTREAM-MOVED.** His gives CLOVER_MOE, MINIMAX_01 and ZONOS3
  upstream's arithmetic filters (`:3888-3894`) and keeps the fork's tensor-keyed filters for QWEN3NEXT / QWEN35 /
  QWEN35MOE / QWEN4EXP (`:3895-3941`). Ours puts CLOVER_MOE in the tensor-keyed branch and gives MINIMAX_01 no branch
  (default filters). For a complete model file both select the same layers: CLOVER_MOE has `wo` on its attention layers
  and `ssm_out` on its recurrent ones (`src/models/clover-moe.cpp:86`, `:99`). ZONOS3 is new at b12500.
- **`LLAMA_PLE_SHARD` limited to qwen4exp: EQUIVALENT on the models listed.** Ours added the architecture test because
  upstream's V4.1 keeps its Engram tables in `per_layer_token_embd`. His V4.1 has its own tensor, so his ungated tests
  (`:961`, `:1347`, `:3039`) only matter for gemma3n / gemma4, which also use that tensor: gemma4 with `-sm tensor` and
  `LLAMA_PLE_SHARD=1` would reach `GGML_ASSERT(n_head_ple > 0)` after `:1347`. Opt-in switch, not a model used here, and
  older than both merges.
- **Order of `llama_ple_prefault` and `load_tensors_hadamard`: EQUIVALENT** (`:2664`, `:2669`; the prefault only touches
  pages).

## 7. `src/models/qwen4exp.cpp`, `src/models/delta-net-base.cpp`

- **qwen4exp conv state outside the ring: EQUIVALENT, his is the leaner form.** After the ring write
  (`src/models/qwen4exp.cpp:1459-1477`) his calls upstream's `build_conv_state_store` (`:1479`;
  `src/llama-graph.cpp:4358-4417`), which writes the slots of the last ubatch only. Ours keeps the fork's plane shift
  for parallel contexts (ours `:1485-1523`), which also preserves older planes. Outside the ring both trees bound the
  rollback by `cell.rs_max`, i.e. the last ubatch, so those older planes are never read: ours spends a `cont` and a copy
  of up to `n_rs_seq` planes per conv state per ubatch for nothing. Both callers pass `keep_snapshots = true` (`:1191`,
  `:1562`), so his ignoring that flag changes nothing.
- **delta-net-base: SAME** (two lines reordered).

## 8. `src/models/models.h` (non-DeepSeek), remaining files

- `models.h` outside the DeepSeek structs: SAME (every hunk is in `llama_model_deepseek4`, `llama_model_deepseek41`
  or the DSpark graph).
- For the DeepSeek reader, seen in my files and not judged: `src/models/dflash.cpp:532` (V4.1 DSpark through
  `graph_dsv4`, ours `graph_dsv41`); the Engram CPU pin in `src/llama-model.cpp:3030-3036`; the `GGML_OP_CPY` cast probe
  and `LLAMA_DSV41_ENGRAM_CAST_DEVICE` in `src/llama-model-loader.cpp`; the `engram_*` members in `src/llama-model.h`;
  and his `tests/test-llama-archs.cpp` skipping `deepseek41` (his own GGUF layout has no fixture).

## 9. Collisions with the 20 Exabit commits

- No textual collision in this group: none of the 20 commits touches `common/speculative.cpp`, `src/` or
  `tools/server/`.
- `6226b3dba` (MMVQ 16-column width) also adds `common/speculative-adaptive.h` and
  `tests/test-speculative-adaptive.cpp`. Nothing includes or builds them at `224b73e24` (no reference in any
  CMakeLists or in `speculative.cpp`), and his tree has neither file. They would arrive in his tree as dead files: drop
  them from the carry or wire them deliberately.
- Interface to watch when the fused-norm and GDN-fold commits are carried (ggml reader's files): they change the
  `keep_rs` dispatch in `gated_delta_net.cu`, and his ring gate assumes the CUDA/HIP kernels write every snapshot slot
  (`src/llama-memory-recurrent.cpp:20`). Also his default DSpark path reads the layer inputs through upstream's device
  buffer where ours used the fork's host taps, so the fused residual ADD gets a different extra reader than in our
  acceptance runs. Run DSpark on its default path after the carry.

## 10. What a follow-up pull request could take, by value

1. MTP: zero the carried row before the device upload (1.2). Small, reachable on the `-np 8` serving configuration.
2. LiliCorr: keep it eager or store its features in the replay (5.2).
3. KV-only store: FP8 scale division, and correct the "mirrors exactly" comment (2.2).
4. DFlash: do not arm the accum, or allocate the static taps, when the device buffer is on; time the device inject
   (5.1, 4). Or decide the default by the long-prompt A/B.
5. Optional: `LLAMA_MTP_DEV`, the two context guards (1.1), the acceptance requirement in the ring tests (3.4), removal
   of `ggml_mul_mat_aux` (4).

Worth adopting from him if our line continues: the ring backend gate, the plan assert, the runtime ring query with the
replay checks, `draft_prepare`, the `verify_row0` guard in `accept()`, and upstream's conv store for qwen4exp.
