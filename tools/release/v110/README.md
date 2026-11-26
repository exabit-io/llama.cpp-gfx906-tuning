# /root/rebase-v110 - llama.cpp v1.1.0 cycle (started 2026-11-26)

Record of the cycle, kept as it goes: one row per step with its result. The method is `RELEASE-RUNBOOK.md` in the
tuning repository (`/root/llama.cpp-benchmarking`); what differs from it this time is written here.

Task (2026-11-26): llama.cpp v1.1.0 is out; redo the cycle of PR mxxm-t/mx-llama.cpp#24. mxxm-t updated his master to
b12500 independently of that PR, so how he handled things is to be compared with how we had.

## State found

| What | Value |
|---|---|
| Upstream release | `v1.1.0` = `171527522` = build b12774 (v1.0.0 was `b1b770d`, b12359: 415 commits) |
| mxxm-t master | `557a6068f` (was `cdc81dae4`): his own merges of b12169 (`6192a1c88`) and b12500 (`897e26808`, second parent `aaa807c56`) with fix commits of his own; 274 upstream commits from b12500 to v1.1.0 |
| PR #24 (ours, `merge-v1.0.0` = `a73c601e6`) | open, CONFLICTING, no review, no comment from him: obsolete |
| Exabit commits to carry | 20 (`a73c601e6..224b73e24`): 18 of v1.0.0 + the LLVM 24 tile rule + its threshold 85 |
| Box | rebooted 19:19 UTC 2026-11-26, kernel 7.0.0-43, ROCm 10.1 runtime (10.0 still installed), CPU governor schedutil since 2026-11-22, installed llama deb `-r2` (1.0.0, `224b73e24`) |

## Two things that broke every build before any code was touched

1. **`gcc-14` without `libstdc++-14-dev`** (installed 2026-11-17 09:50 UTC by `apt-get install -y gcc-14`, for a
   kernel-module build). Both ROCm compilers pick the newest GCC directory and then fail at configure with
   `'cmath' file not found`. `build-rocm.sh` now pins the HIP compiler to the HOST C++ compiler's GCC directory
   (`--gcc-install-dir=$(dirname $(c++ -print-libgcc-file-name))`), which keeps the headers every earlier build used
   (GCC 13). The deb build script needs the same (to do).
2. **The server's web UI is downloaded at build time** (upstream, after v1.0.0): from a Hugging Face bucket by build
   number `b<commit count>`. A fork's count names no upstream build (his: b12717, HTTP 404), the script falls back to
   `latest`, and today's `latest` no longer has the files his tree's check requires (`UI: missing required asset(s)`):
   **his master does not build out of the box today.** `build-rocm.sh` takes `UIVER` and exports `HF_UI_VERSION`:
   `UIVER=b12500` for his tree, `UIVER=b12774` for anything on v1.1.0. The package must name it too.

## Steps

| Step | Script / where | Result |
|---|---|---|
| his merge against ours, file by file | `/root/wt-cmp-ours-b12500` (our `a73c601e6` + upstream b12500, uncommitted); `cmp/BRIEF.md`, `cmp/A-backend.md`, `cmp/B-llama-core.md`, `cmp/C-deepseek.md` | running |
| his master built and tested | `/root/wt-mxxm-b12500`, `/root/build-mxxm-b12500` (ROCm 10.0 compiler), `/root/build-mxxm-b12500-101` | running |

## The merge (`/root/wt-merge-v110`, branch `merge-v1.1.0`, on his `557a6068f`)

`git merge v1.1.0`: 19 files with conflicts, 32 hunks (`merge-v110.merge.out`). `resolve.py FILE N HOW` replaces one hunk.

Resolved, mechanical: `.github/actions/windows-setup-cuda/action.yml` (his retry loop + upstream's new `cuda_nvml_dev`
package), `ggml-cuda/CMakeLists.txt` and `ggml-hip/CMakeLists.txt` (upstream's `ggml_cuda_fattn_instances()` + the fork's
`q8_repack/*.cu` glob), `fattn-vec.cuh` (upstream passes the mask stride `s31` from `nb[1]`, which is what the fork's
local `s31` did; the fork's `maskh[j*s31 + ...]` stays), `llama-arch.cpp` (both functions), `top-k.cu` (the fork's
fixed-order gather stays; upstream only renamed a popcount in code the fork replaced), `norm.cu` and `rope.cu` (the
fork's `clamped` template switch AND upstream's new `do_multiply`/`do_add` and `supports_mrope` switches).

**The hard one: upstream #31669 "GDN: use replay instead of rollback" (2026-11-24, two days before the release).**
For the delta-net models (qwen35, qwen35moe, qwen4exp, clover-moe, KDA models) upstream REMOVED the snapshot branch:
with `n_rs_seq > 0` the context keeps one state plus the inputs of the last ubatch, and the GDN op replays the kept
tokens before the new ones (`ggml_gated_delta_net_ext`, `cparams.rs_replay`, `gdn_replay_args`). The fork has its own
answer to the same problem, the snapshot RING (`ring_enabled`: one sequence, verified backends; planes, `plan_snapshot_writes`),
and for parallel contexts it used upstream's plane-indexed snapshots, which upstream has now removed for these models.
Files: `llama-memory-recurrent.{h,cpp}`, `delta-net-base.cpp`, `qwen4exp.cpp`, `llama-graph.{h,cpp}`, `llama-context.cpp`,
`gated_delta_net.cu`, `ggml-cuda.cu`, `ggml-backend.cpp`.
Decision (2026-11-26): keep both, mutually exclusive per context. A ring context stays a ring context (the fork's
behaviour unchanged); every other context with `n_rs_seq > 0` follows upstream (replay), and the fork's GDN kernels
must then cooperate with a replay op the way upstream's own chunked kernel does (replay-only pass, then the fast
kernel; not when `keep`). To be tested: `-np 4` MTP under `-sm tensor` (replay through the meta backend).

Also found: upstream #32901 renamed and regenerated the flash-attention instances
(`template-instances/fattn-tile-instance-f16-f16-dkq256-dv256.cu`): the Exabit LLVM 24 rule names the old file and the
kernel's data loading changed, so the rule is re-ported and RE-READ (tilescan on every kernel, design points, MTP at depth).

Resolved 2026-11-26, the fork-heavy ones (all 19 files; first build `build-merge-v110.log`):

| File | How |
|---|---|
| `ggml-backend.cpp` | the fork's staged input loop stays whole; upstream's `ggml_backend_sched_copy_input()` (#30685, async host inputs) and its prefetch function are not taken (he had already left the prefetch out in his b12500 merge); upstream's scratch-tensor mechanism and `fast_async_host_input` flag merged around it |
| `llama-context.cpp` | the fork's `LLAMA_NO_FUSED_LID` block dropped (upstream removed the unfused indexer graph, #32650); the fork's drain before a reused graph (`LLAMA_REUSE_NO_DRAIN`) KEPT: upstream removed that drain because its new input copy made it unnecessary, and the fork does not use that copy. `cparams.rs_replay` moved BELOW the fork's `n_ubatch` clamp of `n_rs_seq` (upstream sets it before; with the clamp it would stay true at `n_rs_seq` 0), and switched off when `llama_memory_recurrent::ring_supported()` (new static, the fork's ring condition) or `LLAMA_RS_REPLAY=0` |
| `llama-memory-recurrent.{h,cpp}` | both sets of members; ctor: `rs_replay` as asked, the ring only without it; `seq_rm`: the fork's block with upstream's replay branch (`rs_pend -= rollback`) inside it |
| `llama-graph.{h,cpp}` | ring inputs and replay inputs both; `set_input_rep()` rewritten in the fork's style (`ggml_backend_tensor_set`, only allocated inputs) and the narrower conv window made an input of its own instead of a view (the fork's staged-input contract, as for `s_copy_tail`) |
| `delta-net-base.cpp`, `qwen4exp.cpp` | `build_recurrent_attn`: replay when `cparams.rs_replay`, else the fork's snapshot code (ring or plane-indexed); upstream's `GGML_ASSERT(n_rs_seq == 0)` gone |
| `gated_delta_net.cu` | the fork's column-thread and chunked kernels only for ops WITHOUT replay inputs (`if constexpr (!replay)`); a replay op runs on upstream's recurrent kernel. **Cost to measure:** prefill in a replay context no longer uses the chunked kernel |
| `ggml-cuda.cu` | the fork's host-copy branch + upstream's new wait of the source stream on the destination |
| `deepseek41.cpp` | his graph (he kept his own DeepSeek-V4.1 implementation where we had taken upstream's); upstream's one change to the shared loader merged by itself |
| `ggml.h` | `ggml_gated_delta_net(..., K)` not deprecated in the fork (the ring builds it) |
| `tests/test-backend-ops.cpp` | upstream deleted the K > 1 snapshot cases; restored as `test_gated_delta_net_snapshots` (six cases from his tree) |

## His handling against ours, and what the tests say

Three readers compared his b12500 merge with ours file by file (`cmp/A-backend.md`, `cmp/B-llama-core.md`,
`cmp/C-deepseek.md`; `cmp/B2-merge-review.md` is a second read of my v1.1.0 merge). Their claims are from reading; the
ones marked TESTED were run here. His master = `557a6068f`, built by ROCm 10.0's compiler, ROCm 10.0 runtime
(`queue-mxxm.sh`): `test-backend-ops` 20565/20633 on each die, PPL 16K x 6 5.6126 (= our v1.0.0 merge), MTP 154/200,
`test-llama-context` 144/145 (the known KV-replica model), four MTP row paths identical text.

| Topic | His tree | Ours (PR #24) | Verdict |
|---|---|---|---|
| Fused top-k MoE softmax on 64-lane rows | reductions at width 32 skip every expert with index mod 64 >= 32 (his own fused-softmax commit met upstream #31617 in a clean merge) | the same defect once carried to b12500 | **defect in both, TESTED: the 68 failing `TOPK_MOE` cases. Fixed on the PR branch (`63181a17e`): 692/692 on four dies** |
| HIP expert-group ballot (`7334d8c27`), `PAD_REFLECT_1D` (`a73c601e6`) | not needed: upstream fixed both by b12500 (#31617, #32179) | our two extra commits | dropped |
| MMA GDN kernel with the snapshot ring | launched unconditionally, writes only the final state | skipped when K > 1 | ours fixes his; NOT reachable on gfx906 (Vega 20 gets the fallback); NVIDIA Ampere+, CDNA, RDNA3.5 only. Not in the PR branch yet (cannot be tested here) |
| Meta fold of the shared-expert add (`same_lanes`) | fixed (`773721de3`) | old fold | his is better; in the merge by construction |
| Scheduler input prefetch | left out | ported (asserts under `-sm tensor` in one layout) | his is safer; the merge follows him |
| MTP rows (deferred prefill, then device rows) | same sequencing | same | equivalent |
| MTP carried row at position 0 | reset runs AFTER the device upload, so a request from position 0 pairs its first token with the previous request's last row | reset before the upload | ours fixes his (read and confirmed in his code); text unchanged under greedy verify, acceptance can differ; still in the merged tree; not yet tested |
| LiliCorr (and v1.1.0's DSpark Markov head) on the deferred path | replay never stores their context features | LiliCorr kept eager | ours fixes his; masked by the server's 4-token prompt tail, shows with `--ctx-checkpoints 0`; not yet tested |
| KV-only MTP store | no FP8 scale division | has it | latent: needs FP8 KV + a GGUF with scale tensors |
| DeepSeek-V4.1 | his own implementation and GGUF layout; upstream-layout files (the one on this box, unsloth's, his own converter's output) do not load. **TESTED on the merge build, CPU only: `wrong number of tensors; expected 1043, got 1003`**, as the reader predicted | upstream's layout (the v1.0.0 package loads the file) | **a product difference, not a bug to fix in a merge: on his line our DSV4.1 + DSpark tests cannot run, and a 1.1.0 package loses that model. Lead's decision whether master carries upstream's layout (a port of its own: two implementations under one architecture name)** |
| Ring tests, backend gate, `plan_snapshot_writes` assert, `accept()` guard | has them | does not | his is better; in the merge by construction |

## Tests of the merge (`queue-merge.sh`, then `queue-merge2.sh`; ROCm 10.0 compiler and runtime)

| Test | Result |
|---|---|
| `test-backend-ops` x4 | 21803/21871, the same 68 `TOPK_MOE`; **with the two fix commits: 21871/21871 on each die** (`queue-merge2`) |
| PPL 16K x 6, `-sm tensor` | 5.6130 +/- 0.0623 (his master 5.6126): the numbers moved a little between b12500 and v1.1.0; attribution = `numerics.sh` (stock b12500 against stock v1.1.0) |
| `-sm layer -lm dio` 64 tokens | same text as his master, 0 fallbacks |
| MTP `-np 1` (ring) | 150/207 first request, 151/205 after; text differs from his master's after 80 tokens (same cause as the PPL) |
| MTP row paths (`mtp-modes.sh`) | four arms identical text |
| `test-llama-context` | first run: ABORT at `fattn-tile.cuh:1230` on `hy_v4_flash-moe` (new in v1.1.0: key 576, value 512, GQA ratio 2; the tile kernel has 576/512 only for ratios 4 and 16), the 61st of 150 models. **Stock v1.1.0 aborts the same way** (`ctx-one.sh`). With `e1711b3d6`: **148 of 150 models**, the two failures are `gemma4-dense` (cp_h, cp_d, frag) and `mistral-large4-moe` (cp_h, cp_d) = the known KV-replica defect; `rollback` 16/16, `rollback_bound` 16/16, **`replay` 15/15** (upstream's state-level test of the replay, qwen35 and qwen35moe among them), `shared` 25/25 |
| Rollback modes (`rs-modes.sh`, MTP n_max 3) | ring = ring with `LLAMA_RS_REPLAY=0` (identical); no error line, no refused rollback in any arm. **Rule (1) NOT met: replay and snapshots at `-np 4` give different text** (long request and short ones); replay equals the ring's text on the short requests. Open: a rollback fault or only the other GDN kernel (replay ops run on the recurrent kernel, snapshots on the fork's kernels) = `rs-modes2.sh` with the kernel held fixed. Speeds, one session, smoke: prefill 370 (replay) against 431 (snapshots) tok/s, decode 59-61 against 62-63, 11-token prompt 14 against 32-36 tok/s |

Fix commits on the PR branch beside the merge: `63181a17e` (top-k softmax width), `e1711b3d6` (576/512 tile selection).

## Master (`/root/wt-master-v110`, branch `master-v110`): the 20 Exabit commits on the merge

Cherry-picked straight onto the merge (no separate reconcile stage: the conflicts are the same ones). `cp-continue.sh`
continues the series; `cp-*.out` are git's messages. Ports:

| Commit | Port |
|---|---|
| term 02 (norm also emits Q8_1) | upstream made `rms_norm_f32` a template over float / BF16 (#32135): `do_q8` is the last template switch, float rows only (`static_assert`), the launcher guards it with `if constexpr` |
| term 03 (pre-add inside the fused norm) | same; the launcher is now three nested `with_tag()` choices (clamped grid, q8, pre-add) around the add / add-type branches |
| term 04 (GDN producer fold) | kernel takes `rep` and `pf`; the fork's fast kernels only without replay and without fold; **new commit `ggml-cuda: no gated-delta-net producer fold on an op with replay inputs`** (the replay stores the graph's k, gate and beta, which a folded op does not compute); the kernel asserts it |
| term 10, 23 | helper before the kernel; upstream's `GGML_CUDA_FULL_THREAD_MASK`; fixup: the DPP reductions use `ggml_cuda_shfl_xor` (32-bit masks no longer compile on HIP) |
| term 01 | upstream's few-row MMQ for MUL_MAT_ID kept (`!use_mmq_few && ne2 <= MMVQ_MMID_MAX_BATCH_SIZE`); the two unreferenced files it carried (`common/speculative-adaptive.h`, `tests/test-speculative-adaptive.cpp`) dropped |
| term 05 | the GCN rows-per-block branch before upstream's new RDNA1 condition |
| LLVM 24 tile rule | **new commit**: the rule globs `fattn-tile-instance-*dkq256-dv256.cu` and fails the configure when nothing matches (upstream renamed the instance; a property on a missing source is silently ignored). Threshold 85 still to be re-read on the new kernel |

Builds with ROCm 10.0 (`/root/build-master-v110`). Not yet tested.

## To do, in order

1. `queue-merge2.sh` results; `numerics.sh` attribution; `rs-modes2.sh`; decide the default for parallel contexts
   (replay or snapshots) by its result and by speed.
2. Tests for the two speculative findings in his tree (carried MTP row; LiliCorr/DSpark-Markov features) and decide
   whether they go on the PR branch as fix commits.
3. Master suite (`pr-test`, `fold-check`, `ctx-check`, `mtp-modes`, `rs-modes`): expected PPL = the merge's - 0.0033
   as in every cycle (5.6093 against 5.6126), folds on/off.
4. Candidates branch; ROCm 10.1 build; tile kernel re-read (`tilescan.py`, design points, MTP at depth); acceptance.
5. Push `merge-v1.1.0`, PR to mxxm-t (supersedes #24), tags; deb 1.1.0 (the deb script needs the GCC pin and `UIVER`),
   test, publish into the rolling channel, dogfood; docs and memory.

## The head-256 tile kernel after upstream #32901 (static, 2026-11-26; `tile/tile-thr-v110.sh`, `tile/tilescan-thresholds-v110.txt`)

Reloads inside loops per kernel `<ncols1,ncols2>` of `fattn-tile-instance-f16-f16-dkq256-dv256.cu` as master-v110 has it:

| Compiler, option | Kernels with reloads in a loop |
|---|---|
| clang 23 (ROCm 10.0), no rule possible | **18 in `<4,2>`** (the 3-or-4-token kernel: MTP verification; v1.0.0: 1), 18 in `<2,4>`, `<1,8>`; 4 in `<2,1>`; 3 in `<8,4>`, `<4,8>` |
| clang 24 + `max-ilp`, no rule | 13 in `<4,2>`, `<2,4>`, `<1,8>`; 10 in `<8,4>`; 6 in `<4,8>`; **none in `<16,2>`** (prefill; v1.0.0: 16) |
| threshold 95 / 92 / 90 | 17 / 13 / 2 in `<4,2>` (and its two siblings) |
| **88** | none in any of the 17 kernels |
| **85 (what `-r2` ships)** | **19 in `<8,1>`** (v1.0.0: none anywhere) |
| 82 / 80 / 78 | 24 / 22 / 22 in `<8,1>`; 80 and 78 also 3 and 6 in `<32,1>` |
| **75, 70, 60, 50** | none in any kernel |

So the rule's number does not carry over: 85 is no longer clean, 88 and 75 (and below) are, and the kernel changed
enough that the old compiler now spills where the service's MTP path runs. A count is not a timing result: the
candidates (88, 75, no rule) and clang 23 go to the design-point cells and to MTP at depth before the rule is set.
The five kernels the service runs (`<1,2>`, `<2,2>`, `<4,2>`, `<8,2>`, `<16,2>`) are clean at 85 as well.

## Second readings of my own work (`cmp/B2-merge-review.md`, `cmp/A2-backend-review.md`), 2026-11-26

- Ring / replay split: no path found where one mode reaches the other's state; the context and the memory decide from
  the same values. Corrections to my notes: QWEN3NEXT never has `n_rs_seq > 0`, and parallel qwen4exp neither replays
  nor rolls back (the fork clamps `n_rs_seq` to 0 there), so the plane-indexed snapshots are reached only with
  `LLAMA_RS_REPLAY=0`. A session saved in one mode does not load in the other (fails cleanly, the prompt is re-processed).
- **A mis-port in master, found by the reader and fixed:** the rows-per-block cases of term 01 (`calc_rows_per_block`,
  `mmvq.cu`) had applied inside upstream's new ORIN branch, whose tail reads like the GCN one. gfx906 was back on
  upstream's 2 rows at 2..8 columns and 1 at 9..16 (speed only; the 4-stream design point is 4 columns). Moved back,
  ORIN branch identical to upstream again, folded into the term's commit; both master builds redone before any timing.
  A hunk that applies cleanly can still land in a look-alike block: after a port, compare each changed function with
  the old commit's version, not only the conflicts.
- His chunked GDN kernel ignores `keep_rs_t` and writes slot 0 only (his code since the ring commit). The op's contract
  allows that for ubatches longer than K, but his ring copies K slots from every ubatch. Reached only when the
  column-thread kernel declines (head size other than 128, or a non-64-lane device): not by the 128-head models here.
  Cannot be shown with `test-backend-ops` (unwritten slots are allowed there). Reported, not changed.
- The MMA GDN launch is still unguarded for K > 1 in both trees (our v1.0.0 merge had the guard): not gfx906.
- Upstream's new VEGA20 rule sends Q4_K above 7 columns to MMQ: term 01's 16-column MMVQ width now holds for Q8_0 and
  the types upstream leaves on MMVQ.
- Read and found right: the clean merges in the backend files, the meta rule for the replay sources, the `norm.cu`
  ports (all eight switch combinations exist, nothing BF16 instantiates a float-only path), the producer fold, the DPP
  reductions, the CMake rules.

## Did the numbers move by themselves? (`numerics.sh`, Qwen3.8-27B Q8_0, PPL 16K x 6, ROCm 10.0)

| Build | `-sm tensor` | `-sm layer` | 192 greedy tokens |
|---|---|---|---|
| stock b12500 | 5.6186 | 5.6102 | `c7849fbf3441` |
| stock v1.1.0 | 5.6070 | 5.6071 | `b2e470365e80` |
| mxxm-t master (b12500) | 5.6126 | 5.6120 | `b4f3623bf280` |
| the v1.1.0 merge | 5.6130 | 5.6112 | `c7849fbf3441` |

Upstream's own numbers moved between b12500 and v1.1.0 (-0.0116 with `-sm tensor`, -0.0031 with `-sm layer`, other
text). The fork's +0.0004 and -0.0008 are inside that. So this cycle has no "same PPL as the last release" check; what is
checked is the merge against stock at the same release for sanity, and master against the merge (folds on and off).

## Candidates pool (`/root/wt-candidates-v110`, branch `candidates-v110`)

Rebuilt on `merge-v1.1.0` in the pool's order (`cand-continue.sh`): the 15 commits that are patch-identical to a v1.0.0
master commit are master-v110's ports of them, the six pool-only terms (17, 18, 19, 21, 22, 27) and the seven docs
commits are cherry-picked from `5c01f4ffd`, and the replay exclusion of the producer fold is added (the pool has term
04). All applied without a textual conflict; the pool-only hunks landed in the functions they name (`q8_repack/*`,
`qwen35*.cpp`, `ggml_cuda_fattn_tile_get_config_amd`). Not built yet, not tested. Still without the `max-ilp` default
and the tile rule, so it is compiled by ROCm 10.0's compiler. `BRANCHES.md` gets its v1.1.0 commit when the PR number
is known.

## The rollback question, settled (`queue-rs.sh`: `rs-modes2.sh` on the merge, `rs-modes.sh` on his master; quiet host)

All arms `-np 4`, MTP n_max 3, `-sm tensor`, one request at a time; modes read from the log (`-lv 4`).

| Arm | Text, long / short | Prefill, 2,089 tokens | 11-token prompt | Decode | Recurrent memory |
|---|---|---|---|---|---|
| replay (upstream), recurrent kernel | `4893c7466db7` / `992ec9e4d9de` | 343.6 tok/s | 14.3-15.5 tok/s | 59.8-61.2 tok/s | 652.8 MiB |
| snapshots, recurrent kernel (`GGML_GDN_NO_COLTHREAD=1 LLAMA_GDN_NO_CHUNK_SNAPSHOTS=1`) | `4893c7466db7` / `992ec9e4d9de` | 391.3 | 30.1-34.4 | 60.8-61.4 | 2,394 MiB |
| snapshots, the fork's kernels | `6ffd45358f64` / `a84585d1e445` | 402.9 | 32.2-36.4 | 62.0-63.1 | 2,394 MiB |
| no drafter | `4893c7466db7` / `9acec5a1439c` | 477.4 | 33-37 | 42.3-42.4 | 598.5 MiB |
| his master, `-np 4` (snapshots, his kernels) | (other build) | 399.4 | 30.6-34.6 | 61.4-61.5 | |

- **Rule (1) met: replay and snapshots give the same text on all four requests once both run the recurrent kernel.**
  The difference the first session saw is the fork's other GDN kernels (rounding), not a rollback fault. Upstream's
  state-level test agrees (`replay` 15/15, `rollback` 16/16).
- **Cost of replay in this fork** (its GDN kernels do not replay, so a replay op runs on the recurrent kernel, and the
  replay graph does more): prefill -14.7 %, decode -2.5 %, an 11-token prompt at less than half the rate; it keeps
  27 % of the recurrent memory (1.74 GiB less at `-np 4`, n_rs_seq 3).
- **Decision: the fork keeps what it had.** Ring contexts keep the ring; other contexts keep the plane-indexed
  snapshots by default; `LLAMA_RS_REPLAY=1` gives them upstream's replay. A merge does not take 15 % of the fork's
  prefill; the memory is the argument for the switch. Merge commit and message changed accordingly (the first
  version of today had replay as the default and `LLAMA_RS_REPLAY=0` as the switch).

## The carried MTP row (`mtp-carry.sh`: requests A, A, A, B, A on one slot, `cache_prompt` false, n_max 2)

| Build | Fresh slot | Every later A (also after B) | Text |
|---|---|---|---|
| merge | 150/207 accepted | 151/205 | same |
| merge + reset before the upload | 150/207 | 150/207 | same |

The draft of a request depended on the request before it; with the reset ahead of the device upload it does not. Third
fix commit on the PR branch.

## Branch state, 2026-11-26 21:20 UTC (nothing pushed)

`merge-v1.1.0`: merge commit (snapshots default, `LLAMA_RS_REPLAY=1` for replay) + top-k softmax width + 576/512 tile
selection + carried MTP row. `master-v110` = that + 22 commits; `candidates-v110` = that + 29. The builds under test in
`chain1.sh` (master suite, MTP depth) predate the last two changes of the merge branch (default flip, carried row):
the final builds are verified again before anything is pushed (`queue-final.sh`).

## Clock clamp, 2026-11-26 19:43 UTC: every SPEED number of the evening after that is void

All four dies latched at sclk 1000 MHz at about 19:43:35 UTC, 24 minutes after the boot, and stayed there (under full
load 88-92 W per die, below the 125 W cap; perf level high ignored). `fold-check.sh`'s watchdog said so at 21:33; the
watchdog of `pr-test.sh` had not fired on its 3-minute PPL step, and I had read only result lines. First clamped run:
`mtp-modes-mxxm` (19:48). The latch came within seconds of a 56-thread build starting while a GPU test ended (CPU
85 -> 262 W; total DC 768 W, no SMC envelope exceeded in the 5-second samples, cause not established).

- **Valid:** every pass/fail, every perplexity, every text and draft count (none depends on the clock).
- **Void as numbers:** every tok/s after 19:43: the MTP smoke values, `rs-modes*` speeds, `mtp-depth-v110` (stopped by
  its watchdog in its first cell, no data). The rollback DECISION rests on text equality, on the memory sizes (both
  clock-independent) and on a speed ratio measured with every arm equally clamped; the ratio is repeated after the
  reboot (`afterboot.sh`, `rs-modes3.sh`) before a number goes into anything public.
- **Rule broken, now kept:** no builds while the dies are loaded, also for correctness runs (the 2026-09-07 rule is
  about total power). The remaining builds of this evening were done while already latched.
- **Next:** a reboot (the lead's step, see the last section); `afterboot.sh` checks the clock before every job and runs
  the rollback rates, MTP at depth, the design points and the final verifications unattended (about 8 h). If the
  check fails right after boot the latch needs a cold power cycle (shut down, unplug 30 s) and nothing is run.

## Pushed and offered, 2026-11-26 22:20 UTC

`merge-v1.1.0` = `a3aa1c772` pushed to exabit-io/mx-llama.cpp; **PR [mxxm-t/mx-llama.cpp#25](https://github.com/mxxm-t/mx-llama.cpp/pull/25)**
(mergeable, clean; text = `pr-body.md`; speeds given as ratios with the clock caveat); #24 closed with a pointer.
Final verification of that commit (`queue-final.sh`, files `final-merge-*`): 21871/21871 x4, PPL 5.6130, 148/150 fixture
models with the default rollback and with `LLAMA_RS_REPLAY=1`, MTP 150/207 on every request, `rs-modes3.sh`'s four rules
met. Not pushed: `master-v110` (`9a7177d0a`), `candidates-v110` (`8eb35b2d6`), tags, `BRANCHES.md` (after the package
is accepted). Owed on the PR: the rollback rates at full clock (a comment).

## WHERE THIS STOPS (2026-11-26 22:35 UTC) AND WHAT THE NEXT SESSION DOES

**The box has NOT been rebooted and the dies are still clamped at 1000 MHz.** The reboot and the boot hook were left to
the lead (the session's permission system declined both). `afterboot.sh` is ready and does the timing work unattended.
Either of these starts it:

    # a) one-shot boot hook, then reboot (the hook removes itself once the clock check has passed)
    (crontab -l 2>/dev/null | grep -v rebase-v110/afterboot.sh; echo '@reboot PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /bin/bash /root/rebase-v110/afterboot.sh >> /root/rebase-v110/afterboot.out 2>&1') | crontab -
    /root/powerstep/tools/safe-reboot.sh
    # b) reboot (or cold power cycle), then by hand
    SETTLE=5 nohup bash /root/rebase-v110/afterboot.sh > /root/rebase-v110/afterboot.out 2>&1 &


1. Read `afterboot.progress`.
   - `CLAMP right after boot`: this boot did not clear the latch. Shut down, unplug for 30 s, boot: the `@reboot` line
     is still there and tries again. Nothing else to do first.
   - `CLAMP ... before <job>`: it latched again mid-queue. Read that job's watchdog log for what ran beside it, cold
     power cycle, then `AGAIN=1 SETTLE=5 nohup bash /root/rebase-v110/afterboot.sh > afterboot2.out 2>&1 &` (finished
     jobs are skipped by their `.afterboot-done-NAME` markers).
   - `AFTERBOOT DONE`: results are in `afterboot-rs-modes3/progress` (the rollback RATES at full clock), `mtp-depth-v110b.progress`,
     `dp-v110.progress` with `screen-dpv110m.tsv` / `screen-dpv110s.tsv`, then `final-masterb-*` (master with the fold
     check), `final-mergeb-*`, `final-candb-*` (queue-final; its Flash-Next step is skipped in this queue). Check the clock column of
     every `*/watchdog.out` before reading a rate.
2. PR #25: post the rollback rates as a comment (the text promises them).
3. Tile rule: apply the rules in the headers of `mtp-depth-v110.sh` and `dp-v110.sh`; put the value (or the removal
   of the rule) into master-v110's "LLVM 24 ... finds its instance by pattern" commit with the new kernel's table in
   the comment; rebuild with ROCm 10.1, check the flag reaches exactly the one object, `tilescan.py` on it.
4. Package: clean detached worktree of the final master (`/root/wt-deb-v110`), `ROCMSTREAM=10.1
   /root/mx-llama-deb/build-mx-llama-deb.sh` (it has the GCC pin and sets `HF_UI_VERSION=b12774`); test before
   publishing as for `-r2` (`deb-test-llama-101.sh`), accept on the installed deb against the ROCm 10.0 source build
   at the design points and against the `-r2` library at MTP depth (`rocm-ab5.sh`, `mtp-accept.sh` of
   `tools/rocm101/`, adapted to `/root/build-master-v110`); a design-point metric more than 2 % under v1.0.0 goes to
   the lead before publishing (rule (3) of `dp-v110.sh`).
5. Fork: push `master` = `gfx906-single` = `gfx906-multi` and `gfx906-candidates` with a lease each, tags
   `gfx906/v1.1.0/r0/{merge,master,candidates}`, the `BRANCHES.md` commit (names PR #25); delete `wip/*`.
6. Publish into the rolling channel, dogfood from it, release notes (say that DeepSeek-V4.1 files in upstream's layout
   do not load in 1.1.0 and that 1.0.0 stays in the channel), report part 2, runbook section 0, `CLAUDE.md`, memory.
7. **For the lead:** whether master should carry upstream's DeepSeek-V4.1 layout beside mxxm-t's (a port of its own).

## Direction from the lead, 2026-11-26: our own project, not bound to his fork

None of the merges offered to mxxm-t was taken (#17, #22, #23, #24; #25 is open). The lead, in two messages:
the fork is Exabit's own project; it follows upstream llama.cpp and pulls in the code that serves its goals. His
gfx906 patches STAY and his later changes keep being looked at and pulled on their merit; what ends is being bound to
his tree (waiting on it, resolving in his favour to stay mergeable, offering merges). Condition named by the lead:
no major, statistically significant performance penalty in following upstream against his fork.

- **The condition is a measurement:** `substrate-v110.sh` (header = arms, cells, statistic and the reading, written
  before the first run), with `sub-build.sh` (builds with RCCL on: `/root/build-sub-{mxxm,merge}`; the third arm is
  `/root/build-master-v110`), `sub-mtp.sh` (the MTP cell) and `sub-stats.py` (paired by block, Student t, Holm, 95 %
  intervals; `LINE=<percent>` for another "major" line than 2 %). Read on the merge against his master (the same
  patches on the newer upstream) and on our master against his master. Upstream alone is NOT a candidate; it is an
  optional arm (`ARMS="stock mxxm merge master"`, `WITH_STOCK=1 bash sub-build.sh`).
  `TRY=1 bash substrate-v110.sh` is the functional pass on a clamped box (no rate of it is a number).
- It is job 3 of `afterboot.sh` (after the rollback rates and its own builds), about 6.5 hours of the 15.
- **What "not bound" changes in this cycle:** where the merge took his handling only to stay mergeable for him, our
  master may now differ. First case: DeepSeek-V4.1 files in upstream's layout (the one on this box, the ones on the
  hub) load in our master again, as in 1.0.0 (work item, see below).
- No further PR to him unless the lead asks; #25 is left as it is.
- Wrong reading of mine, corrected by the lead the same evening: "follow upstream" never meant dropping his patches.
