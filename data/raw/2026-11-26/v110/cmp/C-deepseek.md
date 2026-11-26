# C: DeepSeek-V4, DeepSeek-V4.1 and the DSpark drafter

Read-only comparison of his master `557a6068f` (worktree `/root/wt-mxxm-b12500`) with ours (`a73c601e6` carried to
b12500, `/root/wt-cmp-ours-b12500`). `file:line` is his tree at `557a6068f` unless it says "ours". Nothing was built or
run; the only tool run was a GGUF header reader on the CPU (`gguf-py` from his tree) and two static replays of loader
logic against the header, both described where used.

## Short answer

He kept the fork's own `deepseek41` implementation at both of his merges and dropped upstream's
(merge message of `897e26808`: "V4.1 keeps running through graph_dsv4"). His tree and ours each load exactly one
GGUF layout, and they are different layouts:

| | his `557a6068f` | ours |
|---|---|---|
| `deepseek41` layout loaded | the fork's only | upstream's only |
| the file on this box (upstream layout) | does not load | loads (measured, v0.9.0 and v1.0.0 reports) |
| a fork-layout file (his MXFP4 file) | loads | does not load |
| V4.1 sparse attention (indexer top-k 512, candidate blocks) | not implemented, attends every compressed row | upstream's |
| in-tree converter | upstream's `conversion/deepseek41.py`, which writes the layout his loader rejects | same file, matches the loader |

Verdict counts: OURS-FIXES-HIS 5 (sections 1, 2, 3b, and two minor ones in 4b and 5), HIS-BETTER 1 (minor, 4a),
EQUIVALENT 6, SAME 3, UNSURE 3, UPSTREAM-MOVED 0. Details below, most important first.

---

## 1. GGUF layout of `deepseek41` (questions 1 and 2)

Files: `src/models/deepseek41.cpp`, `src/llama-arch.{cpp,h}`, `src/llama-hparams.h`, the DeepSeek part of `src/models/models.h`.

### What each tree reads

His loader (`src/models/deepseek41.cpp:21-191`) reads only the fork's names:

- keys, all optional: `deepseek41.attention.kv_source_layer_ids`, `.attention.index_source_layer_ids`,
  `.attention.candidate_source_layer_id`, `.attention.candidate_block_size`, `.attention.candidate_topk_blocks`
  (`deepseek41.cpp:27-31`); `deepseek41.engram.layer_ids`, `.num_embeddings`, `.primes`, `.offsets`, `.multipliers`,
  `.token_map`, `.head_dim`, `.n_heads`, `.max_ngram_size`, `.pad_token_id`, `.compressed_vocab_size`
  (`deepseek41.cpp:61-72`; name strings at `src/llama-arch.cpp:335-350`)
- tensors: on KV source layers `blk.N.attn_compressor_kv`, `attn_compressor_norm`, optional `attn_compressor_gate`,
  `blk.N.indexer_compressor_kv`, `indexer_compressor_norm` (`deepseek41.cpp:141-150`); on index source layers
  `blk.N.indexer.proj`, `indexer.attn_q_b` (`:152-155`); per Engram layer `blk.N.engram_embed` (lazy),
  `engram_wkv`, `engram_q`, `engram_k` (`:186-189`; names at `src/llama-arch.cpp:727-730`)

Ours is upstream's loader, byte for byte (`git diff aaa807c56 -- src/models/deepseek41.cpp` is empty in our worktree):

- keys: `deepseek41.attention.kv_source_layers` and `.attention.indexer.source_layers`, both REQUIRED
  (ours `deepseek41.cpp:34-35`); `.attention.indexer.candidate_source_layer`, `.candidate_top_k_blocks`,
  `.candidate_block_size` (`:68-71`); Engram through the PLE keys `deepseek41.ple.layers`, `.ple.ngram_size`,
  `.ple.heads_per_ngram`, `.embedding_length_per_layer_input`, `.ple.layer_multipliers`, `.ple.head_offsets`,
  `.ple.head_vocab_sizes`, plus `.engram.pad_token_id`, `.engram.token_map` (`:86-98`)
- tensors: one `per_layer_token_embd.weight` for all Engram tables (`:155-163`), `blk.N.indexer.attn_k`,
  `indexer.k_norm` on KV sources (`:192-193`), `blk.N.engram_kv`, `engram_gate` (`:201-202`)

Neither loader has a branch for the other layout, so there is nothing that "tells the two apart". The enumerators and
name strings of upstream's layout are still in his `llama-arch.cpp` (`:356-360`, `:912-913`), but nothing in his
`deepseek41` loader or graph reads them; the only reader left is upstream's `llama-model-saver.cpp:360-381`, which
writes them from `hparams.dsv41_kv_src` / `dsv41_topk_src` / `engram_pad_id` (`src/llama-hparams.h:444-449,470`),
fields his loader never fills. His `models.h:1286` also keeps a fork `engram_token_map` member on
`llama_model_deepseek4` that shadows upstream's `llama_model::engram_token_map` (`src/llama-model.h:821`), so the
saver sees an empty map. Consequence: `llama_model_save_to_file` of a V4.1 model in his tree writes a file his own
loader then rejects. Low severity, same root cause.

### Would a GGUF from upstream's converter (b12500 or v1.1.0) load in his tree? No.

Upstream's converter did not change between b12500 and v1.1.0 (`git diff aaa807c56 v1.1.0 -- conversion/deepseek41.py`
is empty), and his tree ships that same file unmodified (`git diff aaa807c56 557a6068f -- conversion gguf-py` shows
only `NEXTN_FC_HIDDEN`). No converter for the fork's layout exists in his tree or in its history: no fork-side commit
ever touched a Python file for it (`git log 557a6068f --not v1.1.0 -G"engram|deepseek41|kv_source" -- '*.py'` is
empty). The fork's key names are the checkpoint's own config field names, which upstream's converter reads and
renames (`conversion/deepseek41.py:154`: `add_attention_kv_source_layers(hp["kv_source_layer_ids"])`). So **the
converter his tree ships produces files his tree cannot load**, and how his MXFP4 file was made is not recorded in
the repository.

Failure path, read from the code: every fork key is optional, so with an upstream file `dsv41_n_kv_source = 0` and
`engram_n_layers = 0` (`deepseek41.cpp:56,74-76`); no compressor, indexer or Engram tensor is requested
(`:141,152,172`); `ml.done_getting_tensors()` (`src/llama-model.cpp:2464`) then throws
`wrong number of tensors; expected N, got M` (`src/llama-model-loader.cpp:1526-1528`). The message does not mention
the layout. His `tests/test-llama-archs.cpp:1200` skips `deepseek41` ("the synthetic fixture does not build the Engram
tables and hash keys this architecture requires"): the fixture is upstream's and writes upstream's keys
(`tests/test-llama-archs.cpp:624-632`), so the architecture has no fixture test in his tree. Ours removed the skip.

### Would a fork-layout GGUF load in ours? No.

Both trees share the V4 base loader's required keys (`src/models/deepseek4.cpp:20-55`, same list in ours), so a file
that loads in his passes that part in ours and then stops at ours `deepseek41.cpp:34` with
`array key not found in model: deepseek41.attention.kv_source_layers` (ours `src/llama-model-loader.cpp:313`).
Not verified against a real file: there is no fork-layout GGUF on this box.

### The file on this box

`/root/models/DeepSeek-V4.1-Flash-1017-UD-IQ1_S/*-0000[1-5]-of-00005.gguf` is **upstream's layout**. Header facts
(read with `gguf-py`, no tensor data):

- keys present: `deepseek41.attention.kv_source_layers = [2, 8, 14, 20]`,
  `.attention.indexer.source_layers = [2, 8, 14, 20, 24, 28, 32, 36]`, `.attention.indexer.candidate_source_layer = 20`,
  `.candidate_top_k_blocks = 2048`, `.candidate_block_size = 8`, `.attention.indexer.top_k = 512`,
  `.ple.layers = [1, 14]`, `.ple.ngram_size = 4`, `.ple.heads_per_ngram = 8`, `.ple.layer_multipliers` (8),
  `.ple.head_offsets` (48), `.ple.head_vocab_sizes` (48), `.engram.pad_token_id = 2`, `.engram.token_map` (129280)
- fork keys absent: `kv_source_layer_ids`, `index_source_layer_ids`, `candidate_source_layer_id`, `engram.layer_ids`,
  `engram.head_dim`, `engram.n_heads`, `engram.max_ngram_size`, `engram.num_embeddings`, `engram.primes`,
  `engram.offsets`, `engram.multipliers`
- tensors: `per_layer_token_embd.weight` `[256, 768022850]` IQ4_NL (the whole of split 5, 110 GB),
  `blk.{1,14}.engram_kv`, `engram_gate`, `blk.{2,8,14,20}.indexer.attn_k`, `indexer.k_norm`,
  `attn_compressor_kv`, `attn_compressor_norm`, `blk.{2,8,14}.attn_compressor_gate`, 8 x `indexer.attn_q_b`,
  `indexer.proj`; no `engram_embed`, `engram_wkv`, `engram_q`, `engram_k`, `indexer_compressor_*`
- 1043 tensors in total; compress ratios 0,0 then 2 for layers 2-19 and 1 for layers 20-39

Ours accepts it: I replayed upstream's metadata checks (ours `deepseek41.cpp:37-127`) against the header and all pass
(source layers, candidate source, Engram table sizes, row ids below INT32_MAX), and the v0.9.0 and v1.0.0 reports
record it running on our branch (`reports/2026-11-14-v100-rebase.md:99-112`).

His tree rejects it: I replayed his `load_arch_tensors` against the 1043 tensor names. It requests 1003 of them and
none is missing, so the load reaches `done_getting_tensors` and throws
**`wrong number of tensors; expected 1043, got 1003`**. The 40 never requested: `per_layer_token_embd` 1,
`engram_kv` 2, `engram_gate` 2, `attn_compressor_kv` 4, `attn_compressor_norm` 4, `attn_compressor_gate` 3,
`indexer.attn_k` 4, `indexer.k_norm` 4, `indexer.attn_q_b` 8, `indexer.proj` 8. This is a static replay, not a run.

The DSpark sidecar `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` (`general.architecture = dflash`,
`dflash.hyper_connection.single_pass = True`, no `output_hc_*`, 78 tensors) uses a layout both trees load: his makes
the head tensors optional when a Markov head is present (`src/models/dspark.cpp:110-117`), ours keys on `single_pass`
(ours `dspark.cpp:111`). It is useless in his tree on this box because its target does not load.

**Verdict: OURS-FIXES-HIS** for the scenario this box and every public download hit, with the reverse stated plainly.

- Scenario: `llama-server -m DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf` (any flags, any backend) on his
  master: load error above. Same for any GGUF from unsloth or from `convert_hf_to_gguf.py` of his own tree. Reachable
  on gfx906: yes, it is the only DeepSeek file here, so `dsv41-test.sh` and the DSpark rows of `drafters-100.sh`
  cannot run on his master. `/root/wt-merge-v110` as currently resolved keeps his loader
  (`src/models/deepseek41.cpp:27` there), so the same holds for that branch.
- The reverse: ours stops loading the file he measures with (`FEATURES.md:214`, "DeepSeek-V4.1-Flash-1017 MXFP4, ten
  MI50"). He added V4.1 features on top of his implementation after his merge (`04dbc7b25`, DSpark, image input), so
  replacing it again, as PR #24 did, removes something he uses. Our handling is not a superset of his.
- For the pull request the facts point to three options, not to one: (a) keep his, add a load-time message when
  `deepseek41.attention.kv_source_layers` is present and `...kv_source_layer_ids` is not; (b) load both, choosing by
  that same pair of keys (the fields for both already coexist in his `llama-hparams.h:368-449` and
  `llama-model.h:557-560,630-631`; what is missing is upstream's loader branch, upstream's graph and upstream's V4.1
  tiers in the KV cache, which key on `model.arch` where his key on `dsv41_n_kv_source > 0`); (c) upstream's only, as
  in #24. v1.1.0 adds `hy_v4_flash`, which reuses upstream's Engram keys and `per_layer_token_embd`
  (`v1.1.0:src/models/hy-v4-flash.cpp:169-236`), so upstream is building on its layout.

Test: `bash /root/rebase-v110/dsv41-test.sh /root/build-mxxm-b12500 <out>`; step 1 should fail at load with the
message above in `layer.log`. No GPU work is reached.

---

## 2. V4.1 attention: his graph never applies the lightning-indexer selection

Files: `src/models/deepseek4.cpp` (his V4.1 runs inside the V4 graph), `src/models/deepseek41.cpp`.

- His: for a V4.1 layer with a compression ratio the attention is `build_hca_attention(..., v41 && tier_idx)`
  (`deepseek4.cpp:1260-1263`), which concatenates the window keys with ALL visible compressed rows and masks only by
  visibility (`:821-840`). The top-k path `build_csa_lid_attention` is gated `!v41 && tier_idx` (`:1254-1259`), and
  the compressor/indexer state branch that feeds it is gated the same way (`:1024`). The V4.1 indexer tensors are
  loaded (`deepseek41.cpp:148-149,153-154`) and never referenced by the V4.1 path: their only users are
  `deepseek4.cpp:626,639,1094,1124`, all V4-only. `hparams.dsv41_candidate_src_layer`, `dsv41_candidate_block` and
  `dsv41_candidate_topk` are read at `deepseek41.cpp:29-31` and used nowhere (grep over `src/`).
- Ours (upstream): every compressed layer takes the top-k of its index source layer
  (ours `deepseek41.cpp:531-632`, `ggml_top_k(score, min(n_comp, indexer_top_k))` at `:611`), with the candidate-block
  preselection for layers after the candidate source (`:556-609`), and applies it as a mask or a gather (`:651-744`).

The two agree while a tier holds at most 512 compressed rows (top-k then selects everything): up to about 512 tokens
for the ratio-1 layers 20-39 and 1024 for the ratio-2 layers 2-19. Beyond that his attends over every compressed row
where the model definition (the checkpoint's `indexer.top_k = 512`, `candidate_top_k_blocks = 2048`) attends over 512.

**Verdict: OURS-FIXES-HIS**, certain as a code fact, unmeasured as an output difference.

- Scenario: V4.1 in his tree with a fork-layout file, any split mode, any prompt longer than about 512 tokens; his own
  figure is a 16965-token prompt (`FEATURES.md:193`), where each ratio-1 layer attends about 17 thousand rows instead
  of 512. Result is not the reference computation, and the attention cost grows with context instead of staying
  bounded. No crash. Reachable on gfx906 hardware (his ten MI50), not on this box (no fork-layout file).
- His `FEATURES.md:183-186` lists what V4.1 changes against V4 and does not mention the indexer, which fits the code.
- What I cannot say: how much output quality changes. That needs the same checkpoint in both layouts and a perplexity
  or KLD run at 2K or more context (ours: PPL 2.2164 at 2K x 4 chunks).

---

## 3. `src/llama-kv-cache-dsv4.{cpp,h}` (about 480 lines differ)

Substance: two different V4.1 cache designs, plus one fix of ours he does not have.

### 3a. V4.1 tiers and cross-layer sharing: follows from section 1

- His: the fork's generalisation of the V4 cache. The two ratios come from hparams (`dsv4_ratio_idx`,
  `dsv4_ratio_plain`, filled at `deepseek41.cpp:53-54`); storage is filtered to KV source layers when
  `dsv41_n_kv_source > 0` (`llama-kv-cache-dsv4.cpp:1355-1375`); a layer after a source REUSES the source's cache
  tensor through `reuse_comp` (`:1387-1390`, passed at `:1398,1406`); the V4.1 state has no overlap (`:1418-1437`).
  The lightning-indexer cache and state are still created for the ratio-2 sources (`:1411-1414,1435-1437`) although
  the V4.1 graph never writes or reads them (section 2).
- Ours: upstream's. `is_v41` by architecture (ours `llama-kv-cache-dsv4.cpp:1331`), rows are `[kv | index K]`,
  per-device MIRRORS of a source for layers on another device (`comp_read`, ours `:1357-1382`), mirrors are left out
  of saved state and refilled after a state read (ours `:337-346`, `:1925-1947`), and a batch whose shared tokens have
  different Engram histories is refused (`dsv41_shared_history_ok`, ours `:1479-1545`).
- Effect of his reuse under plain `-sm layer`: a layer on one device reads a cache tensor that lives on another.
  His `FEATURES.md:356-357` describes exactly that cost for `-sm tensor` ("each hop copied the full cache on every token",
  244 MB per token) and fixes it in the meta backend; for `-sm layer` I did not trace the scheduler, so whether the
  whole view is copied per token there is **UNSURE**.

Verdict: part of section 1 (not a separate defect). For V4 (not V4.1) both caches behave the same: with
`dsv41_n_kv_source == 0` his filters, sizes and state shapes reduce to upstream's V4 values (`:1355-1437`).
**EQUIVALENT** for V4.

### 3b. Rollback bound after a prefill ubatch: OURS-FIXES-HIS (contract level, low reach)

Both trees carry the fork's phase-aware gate: an ubatch with at least 64 tokens per sequence, at most one output and no
pending rollback writes no rollback snapshot (his `:717-738`, reserve side `:954-958`; ours factored into
`dsv4_ubatch_skips_snapshots`, ours `:472-493`).

- His: `update_rs_for_ubatches` still records room for that ubatch,
  `rs_max[seq_id] = std::min(n_seq_tokens[seq_id], n_rs_seq)` (`:1814`), so `seq_rm` accepts a rollback into it
  (`:1562-1575`) and the next ubatch restores snapshot planes that this ubatch never wrote (stale or zero).
- Ours: `rs_max[seq_id] = no_snapshots ? 0 : ...` (ours `:1919`), so `seq_rm` returns false.

Scenario: a DeepSeek-V4 or V4.1 context with `n_rs_seq > 0` (any drafter: `common/common.h:423-434`), one ubatch of
64 or more tokens with one output, then `llama_memory_seq_rm(mem, seq, pos_max, -1)` (or up to `n_rs_seq` tokens
back), then decode. His: the partial-block compressor state of that sequence is restored from the wrong position, the
next compressed rows are wrong, silently. Reach: certain through the C API. Through `llama-server` it needs the
last prompt ubatch to hold 64 tokens or more and to be followed at once by a re-evaluation of the last token, which
is what the server does when a request repeats a prompt that is fully cached
(`tools/server/server-context.cpp:4283-4288`, then `:4325`), for example after a request that was cancelled or had
nothing to generate. With the default `--ctx-checkpoints 32` the server processes the last 4 prompt tokens as their
own batch (`:4481-4498`), which is below the gate, so the default server does not reach it. With
`--ctx-checkpoints 0`, or a task type that makes no checkpoints (`:4361`), I see nothing that prevents it; I read
that path, I did not run it.
Honest caveat on ours: the server's wrapper aborts when the memory refuses
(`common/common.cpp:1632-1636`, same in both trees and upstream), so in that same corner ours turns a silent wrong
restore into `failed to remove sequence`, not into a graceful reprocess. Reachable on gfx906: yes in principle (V4
with DSpark on his box); not on this box in his tree (no loadable DSV4-family file).

Test: the rollback cases of `tests/llama-context/cases-rs.cpp` stay below the gate (`replay`: 20-token prompt at
`n_ubatch 16`, `:34-36`; `rollback`: 9 tokens, `:237`; `rollback_bound`: ubatches of 1, 2, 9 and 10 tokens, `:425`), so the
suite cannot tell the two trees apart here. A case with `n_prompt >= 64` in one ubatch, one output, rollback 3,
compared against a reference context, on the synthetic `deepseek4` fixture, shows it; `LLAMA_DSV4_NO_PREFILL_SNAPSHOT=0`
(his `:718`) is the A/B switch that makes his correct.

### 3c. The rest

- Replica sync after a state read, per-sequence replica clearing, the replica accessors on the comp context: in both
  (his `:418-420`, `:1836-1838`, `:2085-2095`; identical lines in ours). **SAME**.
- The gate itself: same conditions and constants, ours only moved into helpers. **SAME**.
- Block-count padding: his pads every non-overlap tier to `ceil(tokens/ratio)` blocks when a spare slot exists
  (`:618-680`); ours uses upstream's `pad_blocks`, on for the CSA and LID slots and off for the HCA slot, where only
  the "no block completed" dummy remains (ours `:616,671-695`, call sites `:2288-2296`). For V4.1 both keep a verify
  batch's graph shape fixed at ratio 2; at ratio 1 the count is the token count. For V4 the only difference is the
  ratio-128 tier with an ubatch above 128 tokens that is not a multiple of 128, where his avoids a graph rebuild.
  **EQUIVALENT** in results.

---

## 4. DSpark (`src/models/dspark.cpp`, `src/models/dflash.cpp`, `common/speculative.cpp` for the feature paths)

### 4a. Decoder graph

- His: deletes upstream's `graph_dsv41` and sends `BACKBONE_DSV41` to `graph_dsv4` (`src/models/dflash.cpp:530-532`),
  which switches to the shifted hyper-connection mix when the sidecar has no `output_hc_*` tensors
  (`dspark.cpp:405-444`, fold at `:477`, `deepseek4.cpp:343-349`), and skips the per-head query norm for that case
  (`deepseek4.cpp:948-951`). `hyper_connection.single_pass` is still read and still adds 1 to the window
  (`dspark.cpp:25,39-43`).
- Ours: upstream's `graph_dsv41` (ours `dspark.cpp:486-682`), selected by `single_pass`.

Compared line by line the two build the same block: same injection, same RoPE (`dsv4_rope_attn_factor(1, 0) = 1`,
`models.h:1273-1276`), same lagged mix and final fold, no query norm, same MoE call. One difference:
upstream sets `self_attn_mask_swa.block_uniform = true` (ours `dspark.cpp:560`) so every query of a draft block uses the
anchor's mask row; his `graph_dsv4` does not (the flag is set only for Mistral Large 4, `dspark.cpp:594`). With the
standard window rule (`src/llama-kv-cache.cpp:1887,1981-1986`) his draft position k then sees k fewer of the oldest
ring keys than upstream's (window 129 keyed from position P+k instead of P). Upstream states the anchor's window is the
intended one (ours `dspark.cpp:40-41,481-484`) and repeats the convention for the base backbone in v1.1.0.

**Verdict: UNSURE** which matches DeepSeek's reference (not on this box). Effect, if upstream is right: slightly less
context for draft positions 2..n, so possibly lower acceptance there; no crash. To decide: the reference DSpark
inference code, or an acceptance A/B of the same target and sidecar with and without that one line, which needs a
target his tree loads. Loader side: **HIS-BETTER (minor)**: his also loads a V4.1 sidecar that has no
`single_pass` key (`dspark.cpp:110-117`), ours requires the key or the head tensors (ours `dspark.cpp:111-115`); for
the sidecar on this box both load.

### 4b. Feature paths: deferred prompt encode, device buffer, host copies

Both trees have the same two switches, `LLAMA_DFLASH_DEV` and `LLAMA_DFLASH_DEFER`. The order differs.

- Ours (`a73c601e6:common/speculative.cpp:1206-1243`): the deferred encode is decided first (`:1217-1226`), and the
  device buffer is used only when it is off (`:1234`). Default = deferred prompt encode; `LLAMA_DFLASH_DEFER=0` =
  device buffer; with `LLAMA_DFLASH_DEV=0` too = host copies. Measured on this box: same text and acceptance on all
  three, 20.1 / 21.1 / 20.5 tok/s (`reports/2026-11-14-v100-rebase.md:109-111`).
- His (`common/speculative.cpp:1186-1224`): the device buffer is decided first and on its own (`:1192`), then the
  accumulation buffer for the deferred encode is allocated as well (`:1217-1223`). In device-buffer mode the target
  skips the host extraction that fills that buffer (`src/llama-context.cpp:3234-3236`, the only writer of the row
  epochs is `:3392`), so `llama_layer_inp_accum_span_epoch` returns 0 (`src/llama-context.cpp:1955-1961`), the
  deferral test `if (epoch != 0)` (`common/speculative.cpp:1498-1500`) never passes, and every chunk takes the
  device-buffer inject (`:1559-1574`). Default = upstream's device buffer whenever the draft can read it
  (`src/llama-context.cpp:1598-1618`); the fork's deferred encode runs only with `LLAMA_DFLASH_DEV=0`.

No wrong output follows (the `epoch != 0` guard is what prevents the zero-row replay we excluded the combination
for). Two side effects in his: the buffer stays allocated and unused, `taps x n_seq x n_ctx_seq x n_embd x 4` bytes of
pinned host memory (`src/llama-context.cpp:1896-1909`): 480 MiB for V4.1 at `-c 8192`, 1.9 GiB at `-c 32768`; and the
log says "deferred prompt encode enabled" while it never engages. His merge message ("every DFlash head shares the
deferred flush") suggests he believes it is active.

**Verdict: EQUIVALENT** in output (expected; measured only in ours), **OURS-FIXES-HIS (minor)** for the unused pinned
allocation and the misleading log. Scenario: any DFlash, DFlash2 or DSpark drafter on the same device as the target's
output, default environment. Reachable on gfx906: yes. Which default is faster is not settled: on this box the device
buffer decoded 2 to 5 % faster than the deferred path on two short prompts; the deferred path exists for long-prompt time to
first token, which nobody has measured against the device buffer.

Test: start a DSpark or DFlash server on his build and read three log lines, `DFlash target features via ...`,
`deferred prompt encode ...`, `layer-inp accum: ... MiB pinned`; A/B with `LLAMA_DFLASH_DEV=0` on a 16K prompt.
Needs a loadable target in his tree, which for DSV4-family models this box does not have.

Side finding outside my files, for whoever owns `common/speculative.cpp`: ours keeps LiLiCorr off the deferred path
(`a73c601e6:common/speculative.cpp:1219-1220`) because only the eager inject stores its context features. His shares
`init()` across heads and LiLiCorr only turns the device buffer off (`:2055-2060`), so its prompt chunks of 16 rows or
more are deferred, and the replay (`:1346-1396`) does not call `store_context_features` (`:1639` is the only call).
`draft()` then reads the anchor's slot without checking its position (`:2112-2114`). With the server's 4-token
prompt tail the anchor still comes through the eager path, so the default server is fine; with `--ctx-checkpoints 0`
the first draft after a prompt reads stale or zero features. Read from code only. Reachable on this box
(Nemotron with LiLiCorr).

---

## 5. Engram: hash, gather, image spans

- Hash: same arithmetic in both (his `deepseek41.cpp:270-283`, ours `deepseek41.cpp:269-276`): rolling xor of
  compressed id times per-layer multiplier, modulo the per-head prime, plus the head offset; padding with the
  compressed pad id; look-back stops at the sequence start and at image cells. **EQUIVALENT** for text.
  Two edge differences: his maps an out-of-range token id to compressed id 0 and hashes it (`:240-242`) where
  upstream treats it as dead (ours `:251-253`); his uses `ggml_sgn` for the gate sign, 0 at exactly zero
  (`:417-420`), upstream uses +1 (ours `:397-399`). Neither matters for real token streams.
- Shared tokens: his asserts `n_seq_id == 1` per token and aborts otherwise (`deepseek41.cpp:245-247`); ours resolves a
  shared token by its first sequence and refuses the batch with an error when the histories differ
  (ours `deepseek41.cpp:305-307`, ours `llama-kv-cache-dsv4.cpp:1541-1545`). **OURS-FIXES-HIS (minor)**: a batch with a
  token assigned to several sequences (the `llama-batched` pattern) aborts the process in his. Not used by the server.
- Gather: his reads rows in the graph from per-layer lazily read host tables, pinned to the CPU buffer type
  (`deepseek41.cpp:186,387-389`, `src/llama-model.cpp:3028-3035`), with an opt-in host-thread gather for MXFP4 tables,
  `LLAMA_DSV41_HOST_INPUT_THREADS` (`deepseek41.cpp:287-312,329-359`; his own figure in the subject of commit `5da5264e0`: "16k
  prefill 162 -> 588" in tensor mode). Ours reads rows in the graph from the single lazily read table and issues one batched
  readahead for all Engram layers before the first gather (ours `deepseek41.cpp:308-317,789-803`). Different answers to
  the same page-fault cost, not comparable without a measurement on the same file. **UNSURE** which is faster.
  His `FEATURES.md:200` still calls the gather "not yet solved".
- Diagnostic gates only in his: `LLAMA_DSV41_NO_ENGRAM` (`deepseek4.cpp:1312`), `LLAMA_DSV41_NO_COMPRESS` (`:986`),
  `LLAMA_DSV41_QNORM` (`:945`), `LLAMA_DSV41_ENGRAM_CAST_DEVICE` (`src/llama-model-loader.cpp:1320-1327`). Only in
  ours: `LLAMA_DSV41_SPARSE=0` (ours `deepseek41.cpp:772`).

## 6. Routing and the image routing bias (his commit `04dbc7b25`)

`04dbc7b25` makes his V4.1 loader accept the optional `blk.N.exp_probs_b_vl.bias` (`deepseek41.cpp:159-160`), builds no
Engram for an embedding ubatch (`deepseek4.cpp:1322-1326,1339`) and asserts tokens in the hash input
(`deepseek41.cpp:255-257`). The graph already picked the image bias for an embedding ubatch
(`deepseek4.cpp:1392-1398`). Upstream's V4.1 does the same three things (ours `deepseek41.cpp:207`, `:790`,
`:883-887`). Mixed token/embedding ubatches are disabled for `deepseek41` in both trees
(`src/llama-arch.cpp:1493-1508`), so upstream's per-token `engram_keep` mask is never built in ours either.
**EQUIVALENT**. Before this commit a fork-layout file carrying that bias failed his load with a tensor-count error.

One upstream change he does not carry: the router logits in F32 (`d6be06557`, #31851; ours `deepseek41.cpp:878-881`).
The router weight is F32 in upstream's converter output and in the file on this box, so I expect no numeric
difference on HIP for it; I did not trace the matmul path for that and nothing was measured. The v1.1.0 form of it (`prec_policy.set_acc`) has already been added to
his loader in `/root/wt-merge-v110` (`src/models/deepseek41.cpp:158-159` there), which is the right port.

## 7. Tensor split

- The fork's routing for V4.1 is the same in both trees: attention replicated by architecture
  (`src/llama-arch.cpp:1559-1570`, same in ours) and the design-B head split admitted for `DEEPSEEK41`
  (`src/llama-model.cpp:669`, ours `src/llama-model.cpp:670`). **SAME**.
- What differs is what runs under it. His: the fork implementation, documented scope "`-sm layer`, and
  `-sm tensor -tps 2`" (`FEATURES.md:205`), with the cross-stage cache transfer fixed in the meta backend. Ours:
  upstream's implementation under the same routing, measured on four dies at 30.9 tok/s greedy and coherent
  (`reports/2026-11-14-v100-rebase.md:104`). `LLAMA_PLE_SHARD` is limited to `qwen4exp` in ours
  (ours `src/llama-model.cpp:965`) because upstream's V4.1 tables share that tensor; his needs no such limit because
  his tables are separate tensors.
- Not testable side by side on this box (section 1).

## 8. V4 (`deepseek4` architecture) in `src/models/deepseek4.cpp`

His V4 path is upstream's plus the same two fork changes ours carries: the layer-input taps expanded at the end
(`deepseek4.cpp:1332-1335`) and the plain-tier cache accessed through `build_get_k` / `build_cpy_k` (`:821,1197`).
For a V4 file `v41` is false, `hc_shift` is false (the head tensors are required, `:97-99` with
`models.h:1289`), and every generalised branch reduces to upstream's (`:975-999,1254-1267`). **EQUIVALENT**.
API shape differs: he keeps the out-parameter `build_hc_pre` with three extra fork parameters
(`models.h:1310-1320`) where upstream moved to `hc_pre` / `hc_mixes` structs (#31142). No v1.1.0 code calls
`llama_model_deepseek4::graph::build_hc_pre` from outside these files, so this costs nothing now.

## 9. Per file

| File | His | Ours | Verdict |
|---|---|---|---|
| `src/models/deepseek41.cpp` | fork loader + host Engram input, 435 lines | upstream, 933 lines | sections 1, 2, 5 |
| `src/llama-arch.cpp`, `.h` | upstream tables + 16 fork keys, 4 fork tensors | upstream tables | follows section 1; the two shared keys have the same strings |
| `src/llama-hparams.h` | upstream fields + fork fields (`:368-438`) | upstream fields | follows section 1 |
| `src/models/models.h` (DeepSeek) | fork API, no `graph_dsv41`, no `build_attention41` | upstream API | follows sections 1, 4a |
| `src/models/deepseek4.cpp` | V4 + V4.1 inside the V4 graph | V4 only | V4 EQUIVALENT (section 8); V4.1 sections 2, 6 |
| `src/llama-kv-cache-dsv4.{cpp,h}` | fork tiers, reuse, no `rs_max` fix | upstream tiers, mirrors, `rs_max` fix | section 3 |
| `src/models/dspark.cpp` | `graph_dsv4` for both backbones | upstream | section 4a |
| `FEATURES.md`, `557a6068f` | documents the fork V4.1, DSpark and image input; says nothing about the layout or how to convert | documents upstream's V4.1 and tells users to reconvert | follows section 1 |
| `README.md` | still says "Base: upstream `b11553`" (`README.md:21`) at b12500 | "v1.0.0 (b12359)" | his line is stale |

## 10. Upstream between b12500 and v1.1.0 (question 4)

`git log aaa807c56..v1.1.0` on the six files gives 13 commits. Merge result is from `/root/rebase-v110/merge-v110.merge.out`
(`git merge v1.1.0` on his master): of these files only `deepseek41.cpp` and `llama-arch.cpp` conflict.

| Commit | Touches | On his version | Effort |
|---|---|---|---|
| `e412c6f46` remove fused lightning indexer gating (#32650) | `deepseek4.cpp` V4 `build_lid_top_k`, `deepseek41.cpp` | V4 hunk merges clean and deletes the unfused branch (his `deepseek4.cpp:662-686`); the `deepseek41.cpp` hunks are in upstream's `build_attention41`, which his file does not have: conflict, take his side. Outside my files it removes `cparams.fused_lid`, which the fork's `LLAMA_NO_FUSED_LID` switch sets (`src/llama-context.cpp:788-791`) | small here; the switch is another group's |
| `19bc41dbb` apply `.scale` tensors automatically (#32669) | `deepseek4.cpp` (three arguments dropped from `build_moe_ffn`), `deepseek41.cpp` (router F32 moved to `prec_policy`) | V4 hunk merges clean; any fork-only call with the old arity is a compile error, not a silent break. `deepseek41.cpp`: conflict; his loader needs the one `prec_policy.set_acc(layer.ffn_gate_inp, GGML_PREC_F32)` line by hand, already done in the merge worktree | small |
| `522d6fdd4` read quantized indexer K in place (#32795) | `deepseek41.cpp` | upstream V4.1 only; nothing to port, his V4.1 has no indexer | none, dropped |
| `0a7675746` sticky NaN in recurrent state (#31913) | `deepseek4.cpp` `dsv4_append_zero_row` | merges clean. The fork's own `ggml_scale(ctx0, hca_state_kv, 0.0f)` used as zeros (`deepseek4.cpp:1013`) is the idiom this commit replaces with `ggml_fill`; the merge does not touch it. Harmless at ratio 1 (the pooled value is NaN anyway if the input is) | one line, optional |
| `33aeb17f8` Inkling 2 DSpark (#32218) | `dspark.cpp` +197, `dflash.cpp`, `models.h`, `llama-hparams.h`, arch tables | merges clean; it changes the Markov head, `graph_dspark` and the Mistral graph, not `graph_dsv4`. New parameters of `build_dspark_markov_head` have defaults, so his call (`dspark.cpp:492`) stays valid. His dispatch line survives (`dflash.cpp:625-627` in the merge worktree) | none |
| `bafa0420c` packing for d1 models (#32553) | `llama-kv-cache-dsv4.cpp` one initializer line | merges clean | none |
| `c94333133` hy_v4_flash (#32960) | arch tables, `models.h` | additive. Uses upstream's Engram keys and `llama_model::engram_token_map`, which his tree still has beside the fork's shadow member | none here |
| `98d67c8d5`, `55d323507`, `29b52d398`, `c622f623f`, `87f816e87`, `5a09582d8` (new models, Metal BF16) | arch tables, `llama-hparams.h`, `models.h` | additive; the one `llama-arch.cpp` conflict is two functions added at the same place, keep both | mechanical |

So keeping his implementation makes this merge cheap in these files. The cost is elsewhere: three of the thirteen
commits are upstream V4.1 work that lands in code his tree does not have, as did #31851 before them, and section 2
shows what the fork version lacks. If upstream's V4.1 is taken instead (ours), all three apply as they are.

## 11. The 20 Exabit commits

None of them touches a file of this group (`git diff --stat a73c601e6 224b73e24`: `ggml/` and
`common/speculative-adaptive.h` only). I found no collision specific to his version. Two shapes exist in both
versions alike and are therefore not his-specific: the Engram gate multiplies two `RMS_NORM` outputs
(his `deepseek41.cpp:409-412`, ours `deepseek41.cpp:393`), which the fused RMS_NORM/MUL/ADD matcher will see either
way; and a one-token tensor of V4.1's hyper-connection stream has 20480 elements, just under the default
`GGML_TP_AR_MAX_NE=20481`. Whether an all-reduce ever carries that shape I did not check. Neither was exercised.

## 12. What can be tested on this box

| Claim | How | Needs |
|---|---|---|
| his master rejects the box's V4.1 file | `dsv41-test.sh /root/build-mxxm-b12500 <out>`, step 1, read `layer.log` | nothing else; fails before any GPU allocation |
| rollback bound (3b) | new `test-llama-context` case, `n_prompt >= 64` in one ubatch, `n_rs_seq 8`, rollback 3, against a reference context, `deepseek4` fixture; A/B `LLAMA_DSV4_NO_PREFILL_SNAPSHOT=0` | a test edit; CPU is enough |
| DSpark default path and the unused buffer (4b) | three log lines at server start; `LLAMA_DFLASH_DEV=0` A/B | a target his tree loads: none for DSV4 here; any DFlash-family drafter shows the same init code |
| LiLiCorr first draft (side finding) | Nemotron + LiLiCorr, `--ctx-checkpoints 0`, compare acceptance of the first block with `LLAMA_DFLASH_DEFER=0` | `lilicorr-ab.sh` with one more flag |
| dense against top-k attention (2), `block_uniform` (4a), Engram gather speed (5) | not possible here | the same checkpoint as a fork-layout GGUF, or his converter |

## 13. What I do not know

- How his fork-layout GGUF and sidecar are produced, and whether his sidecar carries `single_pass`.
- The size of the output difference from section 2, and which DSpark mask (4a) matches DeepSeek's reference.
- Whether plain `-sm layer` copies a reused compressed cache across devices per token in his tree (3a).
- Whether the server can reach 3b with default settings through a path I did not read; I traced only the
  prompt-cache path and the speculative rollback path.
