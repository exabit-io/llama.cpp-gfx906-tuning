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

## WHERE THIS STOPS (2026-11-27 01:20 UTC, before the reboot of 01:32) AND WHAT THE NEXT SESSION DOES

**The box has NOT been rebooted and the dies are still clamped at 1000 MHz.** The reboot and the boot hook are the
lead's step. `afterboot.sh` is ready and does the timing work unattended
(about 13 hours: rollback rates, the substrate comparison, MTP at depth, the design points). Either of these starts it:

    # a) one-shot boot hook, then reboot (the hook removes itself once the clock check has passed)
    (crontab -l 2>/dev/null | grep -v rebase-v110/afterboot.sh; echo '@reboot PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /bin/bash /root/rebase-v110/afterboot.sh >> /root/rebase-v110/afterboot.out 2>&1') | crontab -
    /root/powerstep/tools/safe-reboot.sh
    # b) reboot (or cold power cycle), then by hand
    SETTLE=5 nohup bash /root/rebase-v110/afterboot.sh > /root/rebase-v110/afterboot.out 2>&1 &

1. Read `afterboot.progress`.
   - `CLAMP right after boot`: this boot did not clear the latch. Shut down, unplug for 30 s, boot: with (a) the
     `@reboot` line is still there and tries again.
   - `CLAMP ... before <job>`: it latched again mid-queue. Read that job's watchdog log for what ran beside it, cold
     power cycle, then `AGAIN=1 SETTLE=5 nohup bash /root/rebase-v110/afterboot.sh > afterboot2.out 2>&1 &` (finished
     jobs are skipped by their `.afterboot-done-NAME` markers).
   - `AFTERBOOT DONE`: results are in `afterboot-rs-modes3/progress` (the rollback RATES at full clock),
     `sub.progress` (the substrate comparison; its statistics are at the end, `python3 -I sub-stats.py sub` prints
     them again, `LINE=<percent>` reads them against another line than 2 %), `mtp-depth-v110b.progress`,
     `dp-v110.progress` with `screen-dpv110m.tsv` / `screen-dpv110s.tsv`. Check the clock column of every
     `*watchdog.out` before reading a rate.
2. **The lead's condition** (no major, statistically significant penalty in following upstream against his fork):
   read pairs (1) merge-against-mxxm and (2) master-against-mxxm in `sub.progress` by the rule in the header of
   `substrate-v110.sh`, and report the verdict with the effect sizes and intervals.
3. PR #25: post the rollback rates as a comment (the text promises them). No other PR to him unless the lead asks.
4. Tile rule: apply the rules in the headers of `mtp-depth-v110.sh` and `dp-v110.sh`; put the value (or the removal
   of the rule) into master-v110's "LLVM 24 ... finds its instance by pattern" commit with the new kernel's table in
   the comment.
5. Master: squash the two commits of `wip/ds41-upstream-layout` into ONE commit on top (message from the two, with
   the test results below instead of "not built"), apply the port notes (`portnotes-v110.py`), rebuild with ROCm
   10.1, check the tile flag reaches exactly the one object, `tilescan.py`, final suite (`queue-final.sh` with
   FOLD=1, plus `TAG=... bash ds41-check.sh` and `ds41-dspark.sh` on the result).
6. Package: clean detached worktree of the final master (`/root/wt-deb-v110`), `ROCMSTREAM=10.1
   /root/mx-llama-deb/build-mx-llama-deb.sh` (it has the GCC pin and sets `HF_UI_VERSION=b12774`); test before
   publishing as for `-r2` (`deb-test-llama-101.sh`), accept on the installed deb against the ROCm 10.0 source build
   at the design points and against the `-r2` library at MTP depth; a design-point metric more than 2 % under v1.0.0
   goes to the lead before publishing (rule (3) of `dp-v110.sh`).
7. Fork: push `master` = `gfx906-single` = `gfx906-multi` and `gfx906-candidates` with a lease each, tags
   `gfx906/v1.1.0/r0/{merge,master,candidates}`, the `BRANCHES.md` commit (own project; names PR #25 as offered, not
   taken); delete `wip/*`.
8. Publish into the rolling channel, dogfood from it, release notes (DeepSeek-V4.1: both layouts load;
   `LLAMA_RS_REPLAY=1` writes different, equally valid text than the default on master; 1.0.0 stays in the channel),
   report part 2, runbook section 0, `CLAUDE.md`, memory.
9. **For the lead:** the staged test of the clock latch (dies loaded, CPU stepped, PZ7G logged fast): every positive
   costs a reboot or a cold power cycle, so it needs a go (memory note macpro-gpu-clock-clamp, 22:50 block).

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

## Correctness of master and the candidates pool on the final commits (2026-11-26 22:24-23:38 UTC, dies latched: no rate is a number)

`queue-clamped.sh` = `queue-final.sh` with the fold check, clock guard off, on `/root/build-master-v110` (`9a7177d0a`,
tag `masterc`) and `/root/build-candidates-v110` (`8eb35b2d6`, tag `candc`).

| Check | master | candidates |
|---|---|---|
| `test-backend-ops` x4 | 21871 on ROCm0-2, **21870 on ROCm3** (one `SOLVE_TRI` case, see below) | 21871 x4 |
| PPL 16K x 6 | 5.6088 | 5.6088 |
| fold check, ubatch 16: folds on (twice) / off | 5.0705 / 5.0705 / 5.0819 | 5.0689 / 5.0689 / 5.0732 |
| `test-llama-context`, default rollback and `LLAMA_RS_REPLAY=1` | 148 of 150 both (the two KV-replica models) | 148 of 150 both |
| `mtp-carry.sh` | A 156/198 x4, same text | A 154/200 x4, same text |
| `mtp-modes.sh` | four row paths identical | four row paths identical |
| `rs-modes3.sh` | rules 1, 2, 4 met; **rule 3 NOT met** (replay and snapshots-on-the-recurrent-kernel write different text) | the same |

Two things followed up (`master-checks.sh`, reading rule in its header):

- **`SOLVE_TRI(f32, lhs [128,128,4,1], rhs [32,128,4,1])` on ROCm3: error 1.66e-7 against the test's 1.0e-7.** First
  failure of the case in 42 logs of two cycles; master does not touch the solver; random inputs per run. Repeated 12
  times per die on master and on the merge: **0 failures in 96 runs**. None of the rule's three outcomes occurred, so
  it is not classified by the rule: it fits a rare tolerance miss of the test (the size of the miss, the untouched
  code), no die stands out, and it is not proven. A single `SOLVE_TRI` miss near 1e-7 in a later log is this.
- **Rule 3 of `rs-modes3.sh` on master.** With the three folds off (`GGML_CUDA_NORM_Q8=0 GGML_CUDA_ADD_NORM=0
  GGML_CUDA_GDN_PREFUSE=0`) the two arms STILL differ (replay `acd81c983839`/`992ec9e4d9de`, snaprec
  `43983ee2d15b`/`b241815b1928`), so the folds are not the explanation. `rs-attrib.sh`: the merge code built with RCCL
  on (`/root/build-sub-merge`) writes the same text in both arms, with its own environment and with master's exchange
  settings, so RCCL is not it either. That leaves the 22 commits.
  - `rs-bisect.sh` (5 builds, folds off): first bad commit = **term 23, `76a525c37`, the DPP-based warp reductions
    on GCN** (commit 12 of 22; commit 11 writes the merge's text in both arms).
  - `rs-dpp.sh`: master with ONE line added, `#define GGML_GCN_NO_DPP` at the top of `gated_delta_net.cu` (the opt-out
    term 28 gives the flash-attention kernels): folds off, both arms write the merge's text again
    (`4893c7466db7`/`992ec9e4d9de`, all four requests). So the cause is inside the GDN kernel: its replay loop and its
    live loop carry the same source arithmetic (upstream: "so that a replayed state matches bit for bit"), and they
    stop matching bit for bit when the kernel's lane sums are the DPP ones (butterfly order 1,2,4,8,.. instead of
    ..,8,4,2,1). Which instruction differs was not looked up.
  - **Classification: rounding, not a defect.** Master's backend tests pass 17 `GATED_DELTA_NET` cases with replay
    inputs (`n_pend=4`, head sizes 32 and 128) against the CPU on every die, and `test-llama-context` passes
    `rollback` 16/16, `rollback_bound` 16/16, `replay` 15/15 in both modes. With the folds on, master's replay and
    snapshot text differ anyway (the producer fold reaches snapshot ops only and is not bit-exact). So rule 3 of
    `rs-modes3.sh` is a property of the MERGE; master does not have it and does not need it: `LLAMA_RS_REPLAY=1` on
    master writes different, equally valid text than the default. This goes into the release notes.
  - Not taken: opting the GDN kernel out of the DPP sums would give bit-equal replay back at the cost of whatever the
    DPP sums buy that kernel in decode; that needs timing at full clock and is not worth a slot for an opt-in mode.

## DeepSeek-V4.1 in upstream's layout: the port (branch `wip/ds41-upstream-layout`, `/root/wt-ds41`, build `/root/build-ds41`)

Decided from the lead's direction (our own project, the code that serves us): master loads upstream-layout files
again. Written by a subagent without building (commit `c0bd96794` on `9a7177d0a`): upstream v1.1.0's `deepseek41.cpp`
as `src/models/deepseek41-upstream.cpp` (class `llama_model_deepseek41_upstream`, 52 lines away from upstream's file),
chosen in `llama_model_mapping()` when the file has `deepseek41.attention.kv_source_layers` and not the fork's
`kv_source_layer_ids`; upstream's V4.1 tiers as a second branch in the DSV4 KV cache behind `hparams.dsv41_upstream`;
`LLAMA_PLE_SHARD` excluded for this architecture (as in 1.0.0); the fixture skip removed.

`ds41-check.sh` (reading rule in its header), dies latched, correctness only:

| Check | Result |
|---|---|
| build (ROCm 10.0) | 0 errors |
| `test-llama-archs -a deepseek41` | all 6 passed |
| the model on this box, `-sm layer` greedy 64 | loads, coherent |
| PPL 2K x 4, `-sm layer` | 2.2196 +/- 0.0595 (1.0.0 package: 2.2164 +/- 0.0595) |
| `-sm tensor` greedy 64 | coherent |
| server, `-sm layer` | answers |
| `test-llama-context`, 151 models | the two KV-replica models as on master, **and `deepseek41-moe` fails upstream's `comp_padding`** ("compressed-cache graph was rebuilt for 3 tokens and 1 sequences"): the port used the fork's block padding where upstream's plan builder has `pad_blocks`; values are right, the graph is rebuilt where upstream reuses it. Fix asked of the same subagent (second commit) |
| DSpark sidecar | `dsv41-test.sh` step 4 cannot load it (no `-fitt`); `ds41-dspark.sh` runs it with `-fitt 7168` |

**Padding fix** (second commit `b0e7ee43d`, same subagent, then built and run by me, `TAG=ds41b bash ds41-check.sh`):
the plan builder's padding rule became an argument; the upstream-layout tiers get upstream's rule (pad every stream to
`ceil(tokens / ratio)` blocks), the fork's layouts keep theirs. Why it failed: with `n_ctx` 256 the ratio-2 cache has
no spare slot, the fork's rule then never pads, and 3-token decodes complete 1, 2, 1 blocks, so the plan's lists change
length and the graph is rebuilt. Result: build 0 errors; `test-llama-archs -a deepseek41` 6/6; `test-llama-context`
**149 of 151 models, only the two KV-replica ones fail; `deepseek41-moe` and `deepseek4-moe` pass all 18 cases**; the
model's greedy texts and PPL (2.2196) unchanged. DSpark with `-fitt 7168` (`ds41-dspark.sh`): loads, 118 of 217
drafted tokens accepted on the first commit and again on the fix (1.0.0: 115 of 227). The regular suite on the port
build (`ds41b-pr-test`): 21871/21871 x4, PPL 5.6088, MTP 156/198, Flash-Next the same text: identical to master.

**Where it lives:** branch `wip/ds41-upstream-layout` = `master-v110` + 2 commits. `master-v110` stays at `9a7177d0a`
until the after-boot timing has been read (its build directories and tile libraries are that commit; the port touches
`src/` only, not the GPU library, and no Qwen path). At package time: squash the two commits into one on master,
port note in the message, rebuild, final suite.

## Substrate comparison: builds and functional pass done (2026-11-27 00:5x-01:3x UTC, dies latched)

- `sub-build.sh`: `/root/build-sub-mxxm` (his master `557a6068f`) and `/root/build-sub-merge` (`a3aa1c772`), both
  with RCCL on and librccl linked, ROCm 10.0. The third arm is `/root/build-master-v110`.
- `TRY=1 bash substrate-v110.sh` (tag `subtry`, small cells, one block, clock guard off): every arm ran every cell
  (m4, mtp with 5 requests, m8, m16, s1), no failed run, `sub-stats.py` reads the files. No rate of it is a number.
- `sub-exchange.sh` (verbose, untimed): the three arms link librccl, log the same custom-allreduce path and none of
  RCCL's failure warnings. A working RCCL logs nothing, so "no warning + linked + forced by the environment" is the
  evidence; rule (b) of the header is reworded to say so. The session's environment gives every arm the 20481-element
  size gate (master's default; his is 262144), so his arm runs with our tuned gate: the arms differ in code only.
- The real session is `afterboot.sh` job 3 (`substrate`), about 6.5 hours at full clock.

## After the reboot of 2026-11-27 01:32 UTC: the latch is cleared, the timing queue runs

`afterboot.sh` started by hand at 01:34 (no boot hook). Clock check passed (every die at the top state at idle under
perf high); XGMI raised before the first job (`ifl_allreduce` twice: ring both ways pull 152, push 182, sdma 256 GB/s);
SMC key dump at the clean clock `smc-keys-clean-1127-0135.txt` (for diffing against a dump taken while latched: none was
taken on 11-26, take one the next time).

**Rollback rates at full clock** (`afterboot-rs-modes3/progress`, the merge build `a3aa1c772`, one session):

| Arm | long: prefill / decode tok/s | short decode tok/s (n=3) | 11-token prompt tok/s |
|---|---|---|---|
| ring `-np 1` | 556.9 / 62.88 | 73.14 - 73.88 | 34.1 - 39.0 |
| ring + `LLAMA_RS_REPLAY=1` | 549.6 / 62.55 | 73.09 - 73.79 | 33.6 - 38.5 |
| snapshots `-np 4` (default) | 534.6 / 74.06 | 78.75 - 79.31 | 37.5 - 42.8 |
| replay `-np 4` | 449.6 / 70.02 | 75.29 - 75.91 | 16.8 - 17.8 |
| snapshots on the recurrent kernel `-np 4` | 527.2 / 70.31 | 76.27 - 77.10 | 34.6 - 39.7 |

Texts and modes as on 11-26 (rules 1-4 met). Equal-work comparison (same text): replay against snapshots on the
recurrent kernel = prefill -14.7 %, decode -0.4 % (long) / -1.5 % (short, means 75.60 vs 76.72), 11-token prompt 17.5
vs 37.5 tok/s. The ratios of the latched session (about -15 % prefill) stand. The comment for PR #25 is drafted in
`pr25-rates-comment.md` and NOT posted: after the lead's direction it goes out only on the lead's go.

## Substrate comparison, interim at 05:30 UTC (4 of 5 cells in; final table when `1 x 260864` ends)

Session `sub` (substrate-v110.sh), n = 5 per cell, paired by block, clocks never latched (single dies touch 1000 MHz in
4-13 % of loaded samples, never all four, never longer than 25 s; the latch criterion is 2 min).

- Pair (1), merge against mxxm-t's master, the lead's condition: no penalty in any row so far. Decode +0.14 to
  +0.30 % at 4, 8 and 16 slots, prefill level; MTP single stream decode +6.38 % [+6.13, +6.63] (the texts differ:
  requests 2, 4 and 5 accept more of what is drafted on v1.1.0; rule (c), read at the end).
- Pair (3), master against merge, the Exabit commits: prefill +1.7 to +2.5 % everywhere, decode +1.80 % at 4 x 65536,
  BUT decode **-5.25 % [-5.33, -5.17] at 8 x 4096 and -3.24 % [-3.28, -3.20] at 16 x 2048**. Pair (2) has the same
  two rows (-4.97 %, -3.10 % against his master). `-np 8` is the serving recommendation.
- No earlier measurement separates our commits at 8 or 16 slots (reports, data/raw and the v1.0.0 notes searched: the
  commits were binned at the two design points, 4 x 64K and 1 x 255K, and the 09-08 paired run at 8/12/16 slots
  predates all of them). On master the dense MMVQ window is 16 columns (term 01; his tree: 8, then MMQ), the MMVQ
  kernels carry our row/warp parameters, the folds apply up to 16 columns, and every kernel is scheduled max-ilp.

Armed 05:34 UTC: `after-queue.sh 7414` waits for the timing queue to end by itself (about 14:30 UTC), then builds five
cut points of master's history (`attr8-build.sh`, dies idle) and runs `attr8.sh` (7 arms a0..a6 x 3 cells x n = 3, about
2 h; the reading rule is in its header). Nothing on master changes from that run alone. The queue itself is left as it
is: the tile threshold arms share every other kernel, and the design points are measured again on the package.

## Substrate comparison: RESULT (session `sub`, 2026-11-27 01:38-07:13 UTC, full clock)

75 of 75 runs, no failed run, n = 5 per cell, paired by block, Holm over 10 rows per pair, line 2 %. Clocks: no latch
(per cell the longest stretch of any loaded die at 1000 MHz was 3-5 samples = 15-25 s, never all four; DC at most 856 W).
Raw: `screen-sub-{m4,m8,m16,s1}.tsv`, `sub-mtp.tsv`, `sub.progress` (the full table), `sub-stats.py sub`.

**Pair (1), merge against mxxm-t's master = the lead's condition: MET. No MAJOR PENALTY row; the only penalty row is
prefill at 1 x 260864, -0.21 % [-0.25, -0.16].**

| Cell | Decode, merge / mxxm - 1 | Prefill |
|---|---|---|
| 4 x 65536 | +0.17 % [+0.05, +0.29] (not significant after Holm) | +0.00 % [-0.06, +0.07] |
| 8 x 4096 | +0.30 % [+0.20, +0.39] gain | +0.15 % [+0.04, +0.27] (n.s.) |
| 16 x 2048 | +0.14 % [+0.06, +0.23] (n.s.) | +0.18 % [+0.06, +0.30] (n.s.) |
| 1 x 260864 | +0.39 % [+0.22, +0.55] gain | -0.21 % [-0.25, -0.16] penalty |
| MTP, 1 stream, 47.8K tokens | +6.38 % [+6.13, +6.63], see below | +0.14 % [+0.03, +0.25] (n.s.) |

MTP decode, rule (c): the two trees write different text on all five requests, and the rate follows the acceptance.
Per request, merge / mxxm - 1 with accepted/drafted (mxxm -> merge): r1 +0.62 % (343/502 -> 343/502), r2 +4.74 %
(329/540 -> 334/531), r3 +0.11 % (308/609 -> 307/609), r4 +19.45 % (330/540 -> 359/456), r5 +6.39 % (325/556 ->
335/527). Where the acceptance is the same (r1, r3) the trees are level: the +6.4 % is the text, not the code's speed.

**Pair (2), master against mxxm-t's master (what we ship against his fork):** ahead at the design points and in MTP,
behind at 8 and 16 slots in decode.

| Cell | Decode, master / mxxm - 1 | Prefill |
|---|---|---|
| 4 x 65536 | +1.97 % [+1.83, +2.10] | +1.68 % [+1.63, +1.74] |
| 8 x 4096 | **-4.97 % [-5.13, -4.81] MAJOR PENALTY** | +2.69 % [+2.60, +2.77] |
| 16 x 2048 | **-3.10 % [-3.22, -2.98] MAJOR PENALTY** | +2.70 % [+2.55, +2.86] |
| 1 x 260864 | +2.78 % [+2.52, +3.04] | +0.80 % [+0.78, +0.83] |
| MTP, 1 stream | +5.82 % [+5.73, +5.90] (text, as above) | +1.89 % [+1.79, +1.98] |

**Pair (3), master against merge (the Exabit commits):** decode +1.80 % (4 x 65536), **-5.25 % [-5.33, -5.17]
(8 x 4096)**, **-3.24 % [-3.28, -3.20] (16 x 2048)**, +2.39 % (1 x 260864), MTP -0.53 % [-0.81, -0.25] (different
text: r1 -1.42 %, r3 -2.74 % at slightly lower acceptance, r5 +1.47 % at higher); prefill +1.0 to +2.5 % in every cell.
The header's rule names two cases (a row in (1) only, a row in (1) and (2)); this is the third: the rows are in (2) and
(3) and not in (1), so the penalty comes from our own commits and not from the release. Attribution: `attr8.sh`, armed
behind the queue (section above). It is settled before 1.1.0 is packaged; `-np 8` is the serving recommendation.

## Tile threshold at MTP depth on v1.1.0 (`mtp-depth-v110b`, 2026-11-27 07:14-08:36 UTC, n = 3, full clock)

18 of 18 cells ran. 117,740-token prompt, three requests of 768 tokens, `-np 1`, draft-mtp n_max 3, ROCm 10.1.

| Arm | MTP decode tok/s (min-max) | Prefill tok/s (min-max) |
|---|---|---|
| c23 (clang 23 object) | 45.843 (45.753-45.906) | 615.8 (615.3-616.1) |
| tnone (clang 24, no rule) | 45.901 (45.814-45.951) | 592.9 (592.6-593.3) |
| t85 (as shipped in -r2) | 49.057 (49.006-49.134) | 617.8 (617.5-618.2) |
| **t88** | **49.129 (49.042-49.248)** | 617.8 (617.5-618.5) |
| t75 | no valid sample (rule 0), see below | 620.7 (620.1-621.1) |
| old (v1.0.0 master by 10.1, other text) | 49.013 (48.983-49.035) | 618.5 (618.4-618.7) |

Read by the header's rules:
- (0) c23, tnone, t85, t88 write the same text with the same draft counts in every cell (530/709 518/743 479/862).
  **t75 writes another text on requests 2 and 3 in all three blocks** (523/729, 490/829), so its decode samples are
  discarded. On request 1, where its text and counts are the others', it decodes 47.69-47.77 against t88's
  50.88-51.39: about 7 % slower. That an object differing only in a scheduler threshold changes the text while a
  different compiler (c23) does not is not explained and not investigated; it is one more reason not to take 75.
- (1) A threshold is still worth having: t88 against tnone is up on decode by +7.03 % and on prefill by +4.20 %.
  Without any rule clang 24 costs prefill -3.72 % against clang 23 (tnone against c23, down).
- (2) Taken: **88** (t75 has no valid decode sample and loses on the one comparable request). t88 against c23: decode up
  +7.17 %, prefill up +0.33 %, so the ROCm 10.0 stream loses about 7 % MTP decode at depth with this kernel on v1.1.0.
- (3) t85, the shipped number, is level with t88 here (49.057 vs 49.129); it stays out because it is not clean on <8,1>.
- (5) v1.1.0 (t88) against v1.0.0 at depth: prefill level (-0.11 %); decode 49.129 vs 49.013 at 530/709 518/743 479/862
  against 536/693 517/749 477/864 accepted/drafted (other text, not compared by rule).
- (4) pending: the design-point cells (`dp-v110.sh`, running since 08:36 UTC).

## Design points on v1.1.0: the tile threshold confirmed, v1.1.0 level with v1.0.0 (`dp-v110.sh`, 2026-11-27 08:36-12:58 UTC, n = 3, full clock)

30 of 30 runs, no failed run. Clock column: longest run of a busy die at 1000 MHz 4 samples (4 x 65536) and 5 samples
(1 x 260864), never all four dies at once. ROCm 10.1 runtime, every arm.

| Arm | 4 x 65536 decode / slot | 4 x 65536 prefill | 1 x 260864 decode | 1 x 260864 prefill |
|---|---|---|---|---|
| old (v1.0.0 master by 10.1) | 25.0035 | 820.4 | 32.3495 | 418.6 |
| c23o (clang 23 tile object) | 24.9568 | 817.0 | 32.4695 | 414.7 |
| tnone (clang 24, no rule) | 24.9219 | 797.1 | 32.3958 | 392.4 |
| **t88** | **24.9252** | **819.0** | **32.4304** | **416.6** |
| t75 | 24.3920 | 822.4 | 30.2053 | 419.8 |

Read by the header's rules:
- (4) c23o's own samples span 0.15 % / 0.09 % / 0.09 % / 0.02 %: every metric is read.
- (1) **t88 passes**: against c23o -0.13 % and +0.25 % (4 x 65536), -0.12 % and +0.46 % (1 x 260864). tnone fails on
  prefill (-2.43 %, -5.38 %); t75 fails on decode (-2.26 %, -6.97 %), the same loss it showed at MTP depth.
- (2) **88 is confirmed** as the threshold of the tile rule on v1.1.0 (it was 85 on v1.0.0, which is no longer clean on
  the <8,1> instance; section "The head-256 tile kernel after upstream #32901").
- (3) **v1.1.0 (t88) against v1.0.0: neutral.** Decode -0.31 % and prefill -0.16 % at 4 x 65536, decode +0.25 % and
  prefill -0.48 % at 1 x 260864; all within the +-2 % floor. Nothing to report to the lead as an upstream cost.

## Attribution of the 8- and 16-slot decode penalty (`attr8.sh`, 2026-11-27 12:59-14:10 UTC, n = 3, full clock)

The five cut-point builds took five minutes (ccache), the session 65 minutes: 63 of 63 runs, no failed run, longest run
of a busy die at 1000 MHz 3 samples, never all four.

Decode tok/s per slot, step of each group against the one before (LINE = 1 reading):

| Step | Group | 8 x 4096 | 16 x 2048 | 4 x 4096 |
|---|---|---|---|---|
| a1 / a0 | RCCL default, all-reduce size gate + its default, FA_QUANTS=all | **-12.37 % [-12.76, -11.99]** | **-2.07 % [-2.28, -1.86]** | +0.06 % (n.s.) |
| a2 / a1 | the folds (terms 02 03 04 07 08 10 12) | +1.44 % | -0.18 % (n.s.) | +0.94 % (n.s.) |
| a3 / a2 | DPP warp reductions (terms 23 28) | +0.14 % (n.s.) | +0.11 % | +1.04 % |
| a4 / a3 | MMVQ (terms 01 15 05 09) | +1.80 % | **-1.92 % [-2.07, -1.77]** | +0.66 % (n.s.) |
| a5 / a4 | max-ilp scheduler | +4.50 % | +0.66 % | -0.04 % (n.s.) |
| a6 / a5 | tile rules, GDN fold exclusion | -0.03 % (n.s.) | -0.02 % (n.s.) | +0.11 % (n.s.) |
| a6 / a0 | all 22 | **-5.33 % [-5.63, -5.03]** | **-3.40 % [-3.63, -3.17]** | +2.80 % |

Prefill: max-ilp is worth +2.1 to +2.2 % in every cell; the gate group +0.3 to +0.7 %; all other steps level.

Read by the header's rule:
- (0) The finding repeated: a6 against a0 is a MAJOR PENALTY at the 2 % line in both cells (-5.33 %, -3.40 %; the
  substrate session had -5.25 % and -3.24 %).
- (1) Carriers: **a1 at both widths** (-12.37 %, -2.07 %) and **a4 at 16 slots** (-1.92 %). The steps add up exactly
  (log terms: -5.34 % and -3.40 %). At 8 slots the one carrier takes 12.4 % and the other groups give 7 points back.
- (2) a6 against a5 is level in every row, as expected.
- (3) At 4 x 4096 neither carrier loses (a1 level, a4 +0.66 %): both act only above 4 columns, so both can be confined
  without touching what the design points run.

**Mechanism of a1 (from the logs of this session, no further run needed to see it):** every a0 log at 8 and 16 slots
carries `TP custom AllReduce: auto-selected two-shot for 4-rank decode-sized tensors`; no a1..a6 log does (at 4 x 4096
all arms carry it). The merge and mxxm-t's tree have **no `GGML_TP_AR_MAX_NE` at all**: the 20481 exported by
`screen.sh` does nothing there, and they send every message under 262144 elements (51 rows of 5120) through the fork's
custom all-reduce. Since mxxm `19d784ad0` (2026-09-18, "tp: cross to twoshot for the message sizes a verify batch
sends") that path uses its two-shot kernel from 8192 elements at four ranks. Master's default 20481 (`567a2a769`) sends
everything wider than 4 rows to RCCL. The default was set on 2026-09-08/20 from two measurements that no longer apply:
the broadcast kernel alone "loses at 8-16 rows (-8 %)", and "-19 % prefill at 4 x 64K ungated", which was measured on
builds WITHOUT RCCL, where the fork's code applies no threshold at all (its threshold sits behind `s_have_nccl`). With
RCCL built in, his threshold already keeps prefill-size messages off the custom path. So our gate has been costing
12 % decode at 8 slots since his two-shot commit came in with a merge, and nothing measured it because every
acceptance ran at 1 and 4 slots.

**CORRECTION to `substrate-v110.sh` rule (b):** its header says the environment gives every arm the 20481-element
gate. It does not: the mxxm and merge arms have no such variable and ran the fork's 262144 threshold. What this
changes: nothing in pair (1) (merge against mxxm: both arms on the same threshold, so the lead's condition was measured
like with like and stays MET), and nothing at the design points and the MTP cell (4 rows or fewer: custom path in every
arm; prefill 2048 rows: RCCL in every arm). In pairs (2) and (3) at 8 and 16 slots the arms differed in the collective,
which is exactly the penalty found.

**Mechanism of a4 at 16 columns:** of its four commits only term 01 acts there (`MMVQ_MAX_BATCH_SIZE` 8 -> 16: MMVQ in
place of MMQ for 9..16 columns); terms 05 and 09 act at one column, term 15 on MUL_MAT_ID. At 16 columns MMQ's tile is
full, MMVQ pays per column.

Second stage, running since 14:33 UTC: `gate8-build.sh` (master with `MMVQ_MAX_BATCH_SIZE 8`, the one build needed)
and `gate8.sh` (reading rule in its header): master with gate 20481 (`r`) against master with the fork's threshold
(`c`, `GGML_TP_AR_MAX_NE=0`) and the merge (`a0`) at 8, 12, 16, 24, 32, 48 slots, and `c` against the MMQ-from-9 build
(`m`) at 10, 12, 14, 15, 16 slots. 87 runs, about two hours. `screen.sh` gained `ARMENV="arm:VAR=VALUE ..."` for arms
that share a build and differ in a runtime switch.

## The gate and the MMVQ limit: RESULT (`gate8.sh`, 2026-11-27 14:35-16:31 UTC, n = 3, 87 of 87 runs, full clock)

No failed run. Clock column of the nine watchdog logs: longest stretch of a loaded die at 1000 MHz 6 samples; the one
sample with all four at 1000 MHz (14:46:30, `w16`) is the last sample of a run, the load ending inside the 5 s.
Decode per slot in tok/s; `c / r` and the like are the paired statistic (mean of ln, t, 95 % interval, Holm):

| Cell | a0 merge | r gate 20481 | c fork threshold | m = c, MMQ from 9 | c / r | c / m | m / a0 |
|---|---|---|---|---|---|---|---|
| 8 x 4096 | 23.4143 | 22.1754 | 25.3374 | | **+14.26 % [+13.71, +14.81]** | | (c / a0 +8.21 % [+7.83, +8.59]) |
| 10 x 2048 | 13.5418 | | 13.4615 | 13.6006 | | -1.02 % [-1.09, -0.96] | +0.43 % [+0.32, +0.55] |
| 12 x 2048 | 13.0607 | 12.2686 | 12.9620 | 13.1288 | **+5.65 % [+5.50, +5.80]** | -1.27 % [-1.37, -1.17] | +0.52 % [+0.31, +0.73] |
| 14 x 2048 | 12.4935 | | 12.3801 | 12.5727 | | -1.53 % [-1.60, -1.46] | +0.63 % [+0.59, +0.67] |
| 15 x 2048 | 12.3428 | | 12.1978 | 12.3937 | | -1.58 % [-1.79, -1.37] | +0.41 % [+0.23, +0.60] |
| 16 x 2048 | 12.0997 | 11.7129 | 11.9398 | 12.1534 | **+1.94 % [+1.85, +2.03]** | -1.76 % [-2.08, -1.43] | +0.44 % [+0.24, +0.65] |
| 24 x 1024 | 10.8126 | 11.0823 | 10.7707 | | **-2.81 % [-2.97, -2.65]** | | (r / a0 +2.49 % [+1.94, +3.05]) |
| 32 x 1024 | 10.1088 | 10.3443 | 10.1302 | | **-2.07 % [-2.49, -1.65]** | | (r / a0 +2.33 % [+2.00, +2.66]) |
| 48 x 512 | 7.2366 | 7.4756 | 7.0969 | | **-5.07 % [-5.77, -4.35]** | | (r / a0 +3.30 % [+3.21, +3.39]) |

Read by the rule in the script's header:

- **(0) The check holds: the gate is the mechanism.** `c` against `r` is attr8's a1 step turned round: +14.26 %
  (interval above +10 %) at 8 x 4096, +1.94 % (above +1 %) at 16 x 2048. The two-shot line is in every `a0`, `c`, `m`
  log and in no `r` log at 8, 12, 16 slots. It IS in the `r` logs at 24, 32, 48 slots, which the rule did not expect
  and which is no contradiction: the last layer's all-reduce carries only the rows that produce logits, a 2048-token
  prefill batch ends 2 sequences at 1024 tokens per slot and 4 at 512, and 2 to 4 rows (10240 to 20480 elements) are
  under the 20481 gate and over the two-shot kernel's 8192. At 2048 and 4096 tokens per slot a batch ends one
  sequence: one row, broadcast kernel, no line. The pattern fits all six cells; it was not checked further.
- **(1) The gate: W\* = 16.** The custom path PAYS at 8, 12 and 16 rows and COSTS at 24, 32 and 48. The default becomes
  16 rows of 5120 plus one element, **`GGML_TP_AR_MAX_NE` = 81921**. 17 to 23 rows were not timed; they go to RCCL.
  This also says something about the fork's own threshold (262144 at four ranks): on this hive it is too wide from
  somewhere between 17 and 24 rows on; master as shipped (`r`) is 2.3 to 3.3 % ahead of the merge at 24 to 48 slots.
- **(2) Prefill, `c` against `r`: no MAJOR PENALTY row; two small ones.** -0.55 % [-0.92, -0.17] (not significant
  after Holm), **-0.85 % [-1.17, -0.52]**, **-0.60 % [-0.68, -0.53]** at 8, 12, 16 slots; level at 24, 32, 48
  (+0.10 %, -0.06 %, -0.02 %). No prefill message changes path between the arms (2048 rows, RCCL under every gate).
  In seconds the three are 0.15, 0.18, 0.17 per run: a fixed cost, which points at the benchmark's 16-token warm-up
  batch (RCCL under 20481, custom under the fork's threshold, so RCCL's first call falls into the timed prefill).
  That is a guess until `gate9.sh` (B) has tested it.
- **(3) The MMVQ limit: L\* = 8.** `m` is ahead of `c` at every tested width, 1.0 % at 10 columns rising to 1.8 % at
  16. By the rule the limit goes back to 8 (upstream's and his tree's value): MMQ from 9 columns, term 01's widening
  goes; its launch knobs and Q8_0 fast path stay (they act at 1 to 8 columns, where the group is a gain). 9 columns
  were not timed and take 8's verdict. September's measurement (stock b10288: MMVQ-16 ahead at 9 to 16 on four dies)
  does not hold on this tree. **The single-die check planned beside this is dropped:** the rule was "the smaller
  limit of the two topologies", four dies already give the smallest value there is, and with 8 a single die runs
  exactly what the merge runs at 9 to 16 columns.
- **(4) `c` against `a0`.** +8.21 % at 8 slots; -0.6 to -1.3 % at 10 to 16 (15 x 2048 a MAJOR row, -1.17 %
  [-1.26, -1.09]), which the limit of (3) explains: `m` is +0.4 to +0.6 % ahead of `a0` at all five widths. Level at
  24 and 32. **48 x 512: -1.93 % [-2.62, -1.24], a MAJOR row the limit does not explain**: with the custom path at 48
  rows the Exabit commits cost 2 % against the merge. The candidate does not run that combination (48 rows go to RCCL,
  and there master is +3.30 % ahead of the merge), so the carrier is named and not looked for; if a later cycle
  widens the gate past 16 rows it has to be found first.
- **(5) Nothing changed on master from this run.** The candidate: `GGML_TP_AR_MAX_NE` default 81921 where the two-shot
  kernel is in play, `MMVQ_MAX_BATCH_SIZE` 8. On existing builds that is `m` at up to 16 rows and `r` from 24 on:
  against master as shipped (`m` / `r`) **+7.01 % [+6.96, +7.06] at 12 slots, +3.76 % [+3.49, +4.03] at 16**, and
  +14.3 % at 8 (`c` / `r`; the limit does not act at 8 columns).

**Scope of the new default.** The gain is the two-shot kernel's. `tp-allreduce.cu` selects it for small messages
(from 8192 elements) at four ranks and more only; two and three ranks stay on the broadcast kernel, for which the
only measurement is September's (it loses at 8 to 16 rows). So 81921 becomes the default at four ranks and more, and
fewer ranks keep 20481. A build without RCCL sends everything over the gate to upstream's internal all-reduce; what
it gets is `gate9.sh` (C).

`gate9-build.sh` + `gate9.sh` (running since 16:39 UTC, 42 runs, about an hour; reading rule in the header):
(B) 12 x 2048, five blocks, one build, gate 20481 / 81921 / 81920: the last two differ only in the path of the
warm-up batch; (C) a build without RCCL at 8 x 4096, 16 x 2048, 24 x 1024 with gate 20481 / 81921 / none.

## The warm-up batch and the build without RCCL: RESULT (`gate9.sh`, 2026-11-27 16:42-17:39 UTC, 42 of 42 runs, full clock)

No failed run; clock columns clean (longest stretch of a loaded die at 1000 MHz 2 samples).

**(B) The prefill penalty of the wider gate is the benchmark's warm-up batch.** One build (master + MMVQ limit 8),
12 x 2048, five blocks:

| Arm | Decode / slot | Prefill tok/s |
|---|---|---|
| `r` gate 20481 | 12.4028 | 1170.29 |
| `k` gate 81921 (the candidate) | 13.1283 | 1159.90 |
| `w` gate 81920 (warm-up batch on RCCL, decode custom) | 13.1228 | 1169.99 |

`w` against `r` in prefill: -0.03 % [-0.29, +0.24], no difference. `k` against `w`: -0.86 % [-1.06, -0.67], 0.18 s per
run. `k` and `w` differ in nothing but the path of the 16-token warm-up batch, so by rule (B1) the cost is RCCL's
first call falling into the timed prefill: once per process, no prefill rate, nothing for the gate. (B2) decode: `k`
against `r` +5.85 % [+5.71, +5.99], `k` against `w` no difference: 81921 does what the fork's threshold did at 12 rows.

**(C) A build without RCCL gets the same default.** Decode per slot / prefill:

| Cell | `n20` gate 20481 | `n81` gate 81921 | `n0` no gate | n81 / n20 decode | n0 / n20 prefill |
|---|---|---|---|---|---|
| 8 x 4096 | 19.8258 / 872.7 | 25.4469 / 869.6 | 25.4355 / 661.6 | **+28.35 % [+27.91, +28.79]** | -24.19 % |
| 16 x 2048 | 10.8668 / 872.6 | 12.1886 / 869.7 | 12.1940 / 659.8 | **+12.16 % [+11.90, +12.42]** | -24.39 % |
| 24 x 1024 | 9.8891 / 888.1 | 9.8903 / 888.4 | 10.7787 / 673.6 | level | -24.15 % |

(C1) holds (gain at 8 and at 16; prefill of `n81` against `n20` -0.35 % and -0.34 %, 0.13 s per run, the warm-up
batch again; no MAJOR row). (C2) 24 x 1024 level in both metrics. (C3) for the record: September's "ungated costs
prefill" stands on today's code (-24 %), and `n0` against `n81` at 24 rows is +8.98 % [+8.41, +9.55] in decode: **open
item, not pursued:** a build without RCCL would want the custom path wider than 16 rows (the internal all-reduce is
much slower than RCCL); no package is built that way.

**Commits (local, `/root/wt-master-v110`):** `MMVQ_MAX_BATCH_SIZE` 16 -> 8 and the gate default 20481 -> 81921 at four
ranks and more (two and three ranks keep 20481; with or without RCCL), each with its table in the message. Port notes
applied (`portnotes-v110.py`, seven commits; tree unchanged). **Final `master-v110` = `c782c317d`**, 26 commits over
`merge-v1.1.0`, all `Joshua <joshua@exabit.io>`:
the 22 of `9a7177d0a` + tile rule 88 + DeepSeek-V4.1 upstream layout + MMVQ limit + gate default.

**Running since 17:41 UTC:** `final-build-v110.sh` (ROCm 10.0 and 10.1 source builds of `c782c317d` with tests, the
deb from a clean worktree `/root/wt-deb-v110`, then the static checks: tile rule on exactly one object, `tilescan.py`,
`maskcmp.py` package against source build). After it, unattended: `queue-rel-v110.sh c782c317d...` = final suite,
DeepSeek-V4.1, `deb-test-v110.sh` (installs the deb), `tbo-pkg-v110.sh`, `accept-v110.sh` (about 6 h), then
`mtp-accept-v110.sh`. Every rule is in the header of its script. Nothing is pushed.

## Final builds and static checks (`final-build-v110.sh`, 2026-11-27 17:41-17:50 UTC), release queue started 17:51

- `/root/build-final-v110` (ROCm 10.0, 138 s) and `/root/build-final-v110-101` (ROCm 10.1, 126 s) of `c782c317d`: 0
  errors, RCCL on, max-ilp on, tests built.
- Package from the clean worktree `/root/wt-deb-v110`:
  `mx-llama-cpp-gfx906_1.1.0.git20261127.c782c317d.rocm10.1.gfx906.20261018172427-1_amd64.deb`, version
  `1.1.0+git20261127.c782c317d.rocm10.1.gfx906.20261018172427-1`, 32,283,478 bytes; Depends on the ROCm 10.1 runtime,
  rocBLAS and RCCL packages only.
- The LLVM 24 rule (`-amdgpu-vgpr-threshold-percent=88`) is on exactly one compile command, the head-256 tile
  instance, in the ROCm 10.1 source build and in the package build; on none in the ROCm 10.0 build.
- `tilescan.py`: none of the 17 kernels of the tile object has a reload inside a loop, source build and package build
  alike (`final-static/tilescan-final-v110.txt`).
- `maskcmp.py` package build against the ROCm 10.1 source build: 1000 functions differ in address constants only
  (`s_add_u32`, `s_mov_b64`), none in an instruction: the package's GPU code is the source build's.

`queue-rel-v110.sh c782c317d...` runs since 17:51 UTC (`queue-rel.progress`; jobs `final`, `ds41`, `debtest`, `tbo`,
`accept`, `mtp`; about 9 hours). `publish-llama-110.sh` is ready for after it (tag `gfx906/v1.1.0/r0/master`, release
`mx-llama-cpp-1.1.0-gfx906-rocm10.1`, then the rolling channel).

## Release queue, the correctness jobs: RESULT (`queue-rel-v110.sh`, 2026-11-27 17:51-18:53 UTC, all rc=0)

Clocks: the idle check passed before every job; in the five watchdog logs the longest run of samples with a busy die
at 1000 MHz is 2 (the latch is 24). MTP single-stream on the source build 74.1-74.9 tok/s (the suite of `9a7177d0a`
on 2026-11-26 was latched: 61).

| check | final master `c782c317d`, `/root/build-final-v110` (ROCm 10.0 compiler and runtime) | the package (ROCm 10.1), installed from the file |
|---|---|---|
| `test-backend-ops` per die | 21871/21871 x 4 | job `tbo` (below) |
| PPL 27B Q8_0, 16K x 6, `-sm tensor` | 5.6088 +/- 0.06224 (= `9a7177d0a`, bit for bit) | 5.6088 +/- 0.06224 |
| greedy `-sm layer -lm dio`, Flash-Next | coherent, 0 buffered fallbacks | the same |
| MTP n_max 2 | 154/200, every request and every row path the same text (`c0adc4434872`) | 154/200 np1, 156/198 np4 |
| fold check (ubatch 16 = 16 columns and 16 rows, the widest batch on the custom all-reduce and the first on MMQ) | 5.0756 on = 5.0756 on again, 5.0679 off | - |
| `test-llama-archs` | 151 models | - |
| `test-llama-context` | default rollback 150 of 151, `LLAMA_RS_REPLAY=1` 149 of 151 | - |
| `mtp-carry.sh` | A A A B A: the three A after the first and the one after B identical | - |
| `rs-modes3.sh` | rules 1, 2, 4 met, rule 3 not (as on `9a7177d0a`, a property of the merge only) | - |
| DeepSeek-V4.1 (`test-llama-archs -a deepseek41`, `dsv41-test.sh`, `ds41-dspark.sh`) | 6 of 6; PPL 2.2196; `llama-cli` texts `8732c2ea` / `8b8499fe` as before; DSpark 110 of 238 drafted tokens accepted | PPL 2.2196, the same two texts, DSpark 110 of 238 |
| package properties | - | 55 ELF files, all with `DT_RPATH=/opt/mx-llama.cpp/lib:/opt/rocm/core-10.1/lib`; with another build and ROCm 10.0 first in `LD_LIBRARY_PATH`: 0 ggml/llama libraries from outside, 0 of 26 ROCm libraries from outside 10.1; RCCL up; `1.1.0-dev (build 13021, commit c782c317d)` |

Read against the earlier runs, three things differ and each is accounted for:

1. **`test-llama-context` 150 of 151 in the default mode** (always 2 failures before: `gemma4-dense` and
   `mistral-large4-moe`, the KV-replica defect). Not a fix: the fixtures get new random weights on every run (four
   runs, four md5 sums of `mistral-large4-moe.gguf`) and the defect's error depends on them: NMSE at step 1 was
   1.35e-05, 5.6e-03, 1.0e-03 and 1.6e-03 in the failing runs and 3.4e-06 to 9.7e-06 in this one, against the test's
   threshold of 1.0e-05. The v1.0.0 cycle saw the same (2 of 3). The defect is open as before.
2. **Greedy texts of SHORT prompts changed against `9a7177d0a`; texts of long prompts and both perplexities did not.**
   Qwen MTP (11-token prompt): `992ec9e4d9de` (156/198) -> `c0adc4434872` (154/200), the same in every test and row
   path; the 2089-token prompt's texts are identical in all five rollback arms. DeepSeek-V4.1 `llama-server` reply
   (18-token prompt): `3017d228` (192 tokens) -> `7f3c7fee` (71 tokens, EOS), while `llama-cli` on the same prompt
   is unchanged. Cause: batches of 9 to 16 tokens are now computed by another kernel (MMQ, was MMVQ) and their
   all-reduce of 5 to 16 rows by another path (custom, was RCCL), so their last bits differ and a greedy near-tie
   falls the other way. The server evaluates the 18-token prompt as 14 + 4 (`prompt processing, n_tokens = 14` in
   `srv-base.log`), `llama-cli` as 18 at once; the Qwen prompt is 11 tokens. Not checked further (no arm with only
   one of the two changes generated text).
3. **On the ROCm 10.1 runtime the first request of every new server process starts slowly**, and the package test is
   the first job of this queue on that runtime. Package, single stream MTP: the very first request after the install
   39.00 tok/s (101 tokens at 22 tok/s, the other 155 at 75; prompt 870 ms against 359 ms on the source build), the
   first requests of the next servers 55.5 and 58.2 tok/s (source build on 10.0: 67.6 and 68.0), two later requests
   62.8 and 56.1 with the time lost in the last 30 tokens, all others 73.9-74.6 (source 74.1-74.9); np4 79.2-79.7
   with a first request of 71.3. DeepSeek-V4.1 `-sm tensor`, 55 tokens in a new process: 18.5 tok/s against 30.3.
   Nothing in the journal at those seconds (no memory-error soft-offline, no GPU message), no latch, kswapd,
   compaction and THP counters 0. The v1.0.0 cycle has the same in its data without having named it (first request
   54.77 tok/s on 10.1 in `pr-test-101`, 68.3 on 10.0). OPEN until the MTP acceptance is read: `mtp-accept-v110.sh`
   runs the package, two source arms and v1.0.0 on the 10.1 runtime, so its first requests show whether 1.1.0 starts
   slower than 1.0.0; its means are compared only after the slow samples are looked at (the v1.0.0 rule).
   Tool for later: `/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms` counts the time a process's GPU queues were
   evicted (2.5-3.5 s within 8 minutes in each `test-backend-ops` process); a stalled request with no growth there
   is not an eviction.

**Job `tbo` (18:53-19:03 UTC): PASSED.** `test-backend-ops` of the ROCm 10.1 source build with the installed
package's `libggml-hip.so.1.1.1` mapped (`maps-N.txt`), one process per die: 21871/21871 x 4; 5799
`FLASH_ATTN_EXT` cases on a die, 177 of them with head size 256 (the tile object the LLVM 24 rule is on).

**Written for after the queue: `firstreq-v110.sh`** (item 3 above; rules F1-F5 in its header, `CHECK=1` run: every
arm maps its own `libggml-hip` and the runtime it names). Five arms (package, package with
`HIP_ENABLE_DEFERRED_LOADING=0`, the same commit by ROCm 10.1 and by 10.0 from source, v1.0.0 by 10.1), 12 requests
per new server, n=3, about 20 minutes. It says whether the slow start is one-time, whether it belongs to the runtime
or to the package, and whether deferred code loading carries it. By its rules only two outcomes go to the lead before
publishing: the package starting slower than the same code from source, or slow requests that keep coming back in
1.1.0 and not in 1.0.0.

Job `accept` runs since 19:03 UTC (84 runs, about 355 min), then `mtp`.

## Timing acceptance and MTP at depth: RESULT (`accept-v110.sh` 19:03-01:00 UTC, `mtp-accept-v110.sh` 01:00-01:55 UTC, 2026-11-27/28)

Queue ended `QUEUE-REL DONE`, every job rc=0. Acceptance: 84 of 84 runs, 0 failed. Clock columns: no latch (longest
stretch of a loaded die at 1000 MHz 6 samples of 24; `wdscan.py`). Full output: `accept-v110.progress`,
`mtp-accept-v110.progress`.

Per arm, mean of three blocks (decode tok/s per slot / prefill tok/s):

| Cell | `pkg` 1.1.0 package | `s100` same commit, ROCm 10.0 | `old` v1.0.0 (10.1) | `a6` `9a7177d0a` | `a0` merge |
|---|---|---|---|---|---|
| 8 x 4096 | 25.078 / 1153.3 | 25.294 / 1115.7 | 21.558 / 1155.6 | 22.100 / 1140.7 | 23.456 / 1128.4 |
| 12 x 2048 | 13.110 / 1142.5 | 13.110 / 1146.0 | 12.157 / 1153.3 | 12.273 / 1166.2 | 13.057 / 1130.1 |
| 16 x 2048 | 12.149 / 1161.0 | 12.169 / 1158.4 | 11.622 / 1166.2 | 11.708 / 1169.7 | 12.101 / 1138.8 |
| 48 x 512 | | 7.4805 / 1209.8 | | 7.4747 / 1210.5 | 7.2265 / 1185.7 |
| 4 x 65536 | 24.961 / 797.8 | 25.102 / 804.4 | 24.987 / 789.3 | 25.067 / 790.4 | |
| 1 x 260864 | 32.422 / 412.7 | 32.474 / 408.4 | 32.383 / 414.6 | 32.454 / 410.1 | |
| 8 x 65536 | 14.928 / 820.2 | | 13.851 / 821.4 | | |

By the rules of the script header:

- **(P1) met.** Design points, `pkg` against `s100`: no MAJOR PENALTY row. 4 x 65536 decode -0.56 % [-0.83, -0.29]
  (p 0.012, not significant after Holm; the runtime's known 0.3 to 0.5 %), prefill -0.82 % [-2.71, +1.10]; 1 x 260864
  decode -0.16 % [-0.34, +0.02], prefill +1.05 % [-0.59, +2.72].
- **(P2) met.** Design points, `pkg` against `old`: 4 x 65536 decode -0.11 % [-0.21, +0.00], prefill +1.12 %
  [-5.56, +8.28]; 1 x 260864 decode +0.12 % [-0.09, +0.33], prefill -0.46 % [-2.01, +1.11]. No difference shown in
  any row. The prefill intervals are wide: see "pauses" below.
- **(P3)** 8 x 65536, `pkg` against `old`: decode **+7.77 % [+7.57, +7.97]** (14.93 against 13.85 tok/s per slot, both
  over R3.1's 12); prefill -0.14 % [-0.18, -0.10], significant and small (820.2 against 821.4 tok/s). No MAJOR row.
- **(C1) met.** The two-shot line is in 3 of 3 `pkg` and `s100` logs and in 0 of 3 `old` and `a6` logs at 8 x 4096,
  12 x 2048, 16 x 2048 (`a0` has it: the fork's threshold; at 4 x 65536 every arm has it: four rows are under both
  gates).
- **(C2) met.** Decode `s100` against `a6`: +14.46 % [+13.82, +15.09], +6.82 % [+5.42, +8.23], +3.94 % [+3.88, +4.01]
  (lines +10, +5, +2). `s100` against `a0`: no penalty row (+7.84 %, +0.41 % n.s., +0.57 %).
- **(C3) NOT MET by the letter**: prefill of `s100` behind `a6` from the cell means 0.643 s at 8 x 4096, 0.371 s at
  12 x 2048, 0.272 s at 16 x 2048 per run (limit 0.3 s); no MAJOR PENALTY row (-2.21 % [-15.20, +12.77], -1.73 %
  [-3.67, +0.26], -0.96 % [-1.85, -0.07]). By the rule the package stopped here until understood: next section.
- **(C4) met.** 48 x 512, `s100` against `a6`: decode +0.08 % [-0.05, +0.21], prefill -0.06 % [-0.13, +0.01];
  against `a0` +3.52 % and +2.03 %.
- **(C5) met.** Design points, `s100` against `a6`: 4 x 65536 +0.14 % / +1.78 % (n.s.), 1 x 260864 +0.06 % / -0.41 %
  (n.s.).
- **(C6)** `pkg` against `old`, decode: **+16.33 % [+15.59, +17.07]** at 8 x 4096, **+7.84 % [+7.64, +8.04]** at
  12 x 2048, **+4.54 % [+4.46, +4.61]** at 16 x 2048; prefill no difference shown (-0.20 %, -0.94 %, -0.45 %).
  16 slots at 2K are over 12 tok/s per request now (12.15; 1.0.0 11.62).
- For the record, `pkg` against `a6` at 4 x 65536 decode: -0.43 % [-0.59, -0.27], significant after Holm, not
  MAJOR at 1 % (the package is the ROCm 10.1 stream, `a6` a 10.0 build).

**MTP at depth** (117,740-token prompt, 3 requests of 768 tokens, n=3, arms = GPU libraries under the package's
binaries; clock column clean, longest stretch 3):

| Arm | MTP decode tok/s | Prefill tok/s | Accepted / drafted | Text |
|---|---|---|---|---|
| `pkg` (installed library) | 49.173 (49.163-49.184) | 617.5 | 530/709 518/743 479/862 | `31c7130217db` ... |
| `t88` (tile object 88, earlier build) | 49.109 | 617.5 | the same | the same |
| `c23` (clang 23 tile object) | 45.719 | 615.5 | the same | the same |
| `old` (v1.0.0) | 49.021 | 618.3 | 536/693 517/749 477/864 | other (`e0ef79073ee6` ...) |

Rule (0): `pkg`, `t88`, `c23` write the same text with the same draft counts; `old` writes another, its decode is
not compared. Rule (1) **met**: `pkg` against `c23` +7.55 % (line +3.5 %), against `t88` +0.13 %, prefill against
`t88` level. Rule (2): prefill against `old` -0.14 % (line -2 %). The three requests of every `pkg` server read 51.0 /
51.6 / 44.9 tok/s in all three blocks: no slow request in this session.

## (C3) on undisturbed samples, and the pauses of the acceptance session (`c3-rerun-v110.sh`, 2026-11-28 02:06-02:42 UTC)

**Why (C3) read over its limit: the acceptance session's prefill samples were disturbed.** What its logs show:

- Run-to-run prefill spread per arm in a cell: 8 x 4096 up to 7.3 %, 12 x 2048 2.6 %, 16 x 2048 0.9 %, 4 x 65536
  7.4 %, 1 x 260864 2.7 %; 48 x 512 0.03 to 0.17 % and 8 x 65536 0.05 %. In the 22 timing sessions of this cycle
  before it that have three or more runs per arm, the largest per-arm spread was 0.88 % (one arm; every other under
  0.5 %) and the median 0.1 %.
- In the slow runs **the whole pipeline pauses**: all four dies at 0 % busy and idle power (bays 174 W of 620 W) for
  5 to 20 s, CPU-zone power unchanged, 20 to 110 s after the start of the run; several pauses in one run. Example
  `old` b2 at 4 x 65536 (prefill 763.1 against 819.9 tok/s in b3 = 24 s): pauses 20:58:19-24 and 20:58:50-59:05.
- They hit runs of **every arm** (the v1.0.0 build `old` and `a6` as well as `s100` and `pkg`), on both runtimes,
  in time windows (19:03-19:45, 20:25-21:06, 21:36-22:12 UTC) and not outside them. Decode of the same runs is not
  touched (spread 0.04 to 0.3 %).
- Not the cause, checked: clock latch (no stretch over 6 samples), power caps (macpro-fand's log: 125 W on every die,
  PL1 240 W throughout), host load (sar: 96 % idle, run queue 1 to 6, no I/O wait), memory pressure (no reclaim scan,
  106 GB free), huge pages and compaction (THP madvise, every `thp_*` and `compact_*` counter 0, 78 pages migrated
  since boot), memory-error bursts (one correctable error per ten minutes), a leftover GPU process (none).
- **What pauses the pipeline is not known.** It did not come back in the rerun, so the eviction log below could not
  test the first candidate (KFD evicting the process's queues).

**(C3) holds on clean samples.** `s100` against `a6`, five blocks, the acceptance's own script with `TAG=c3`:

| Cell | `s100` decode / prefill | `a6` decode / prefill | Prefill spread `s100` / `a6` | Prefill effect | Seconds per run |
|---|---|---|---|---|---|
| 8 x 4096 | 25.308 / 1166.5 | 22.194 / 1173.9 | 0.19 % / 0.16 % | -0.63 % [-0.66, -0.60] | +0.177 |
| 12 x 2048 | 13.131 / 1159.9 | 12.261 / 1169.9 | 0.10 % / 0.15 % | -0.85 % [-0.96, -0.75] | +0.181 |
| 16 x 2048 | 12.181 / 1162.8 | 11.701 / 1170.7 | 0.11 % / 0.14 % | -0.67 % [-0.75, -0.60] | +0.190 |

By the rule written in the script before the run: (V) all three cells READABLE (spread at most 0.5 %); (C3') the
difference is a constant 0.18 to 0.19 s per run, inside 0 to 0.3 s, no MAJOR PENALTY row: **(C3) met**, and the
acceptance's 0.64 s and 0.37 s are put down to the pauses. It is gate9's warm-up batch (0.18 s there). Decode
confirms (C2) with n=5: +14.03 % [+13.90, +14.16], +7.10 % [+7.04, +7.15], +4.10 % [+3.94, +4.27]. Clock columns
clean (longest stretch 5), no power dip that costs prefill.

**(E) New fact from the eviction log (`evsample.sh`, `evsample-c3.log`):** every one of the 30 processes has exactly
one eviction of its GPU queues, 670 to 800 ms, 6 to 7 s after it appears in KFD, i.e. during the model load, the
same in both arms. No run was slow, so by rule (E) nothing is learned about the pauses. The kernel has function
tracing for every path that can evict (`amdgpu_amdkfd_evict_userptr`, `svm_range_evict`,
`kgd2kfd_schedule_evict_and_restore_process`, `amdkfd_fence_enable_signaling`): **open item**, a trace with stacks
on those during a run that pauses names the trigger.

Tools of this reading (read-only, in this directory): `wdscan.py` (clock column of a watchdog log), `runphase.py` (per run: clock, die power, bay
power in the prefill phase), `dipscan.py` (power dips inside the loaded span of every run of every session).

## First request of a new server: RESULT (`firstreq-v110.sh`, 2026-11-28 02:43-02:56 UTC)

Four arms x three server starts x twelve requests (256 tokens, greedy, MTP, `-np 1 -c 8192`); `firstreq-v110/progress`,
`requests.tsv`. Clock column clean.

| Arm | Load s | Request 1 s | Steady s (median of 5..12) | Extra of request 1 | Lost requests |
|---|---|---|---|---|---|
| `pkg` (installed package, 10.1) | 9.53 | 3.56-3.60 | 3.438-3.448 | +0.13 s | none |
| `s101` (same commit from source, 10.1) | 9.52 | 3.57 | 3.436-3.443 | +0.13 s | none |
| `s100` (same commit from source, 10.0) | 11.54 | 3.53-3.56 | 3.402-3.419 | +0.13 s | none |
| `old` (v1.0.0, 10.1) | 9.51 | 3.59-3.60 | 3.459-3.463 | +0.13 s | none |

By the rules of the header: **(F1) ONE-TIME** in every arm (no request from the fifth on lost in any block). **(F2)**
`s101` level with `s100`, `pkg` level with `s101`: the 10.1 runtime does not start slower and the package adds
nothing. **(F4)** `pkg` level with `old`. **(F5)** no lost request; request 1 has 8 to 12 ms of eviction. Neither
outcome that goes to the lead occurred. **(F3) cannot be read:** with `HIP_ENABLE_DEFERRED_LOADING=0` the server does
not start at all, three of three (`hip_platform.cpp:238: Cannot retrieve Static function, error: 200`): never set it
for these builds.

**So the slow starts of the package test (first request 39 tok/s, then 55 and 58; two later requests 56 and 63) are
not a property of the ROCm 10.1 runtime or of the package.** They fell at 18:36-18:53 UTC, at the start of the
evening's pauses (previous section), and do not reproduce: a new server's first request costs 0.13 s in every arm
(71.7 against 74.4 tok/s). The same holds for what v1.0.0's `pr-test-101` showed (54.8): one sample of an episode,
not a runtime difference. One open item instead of two: the pauses.

New in the data: from the third request on, a server's `evicted_ms` grows by 136 to 174 ms (sum over the four dies)
in every request, in every arm; in 4 of 12 servers it stops after the fifth request. Requests with and without it
differ by 3 to 10 ms of prompt time and not in generation time.

## What evicts the queues: located (`kfdtrace-v110.sh`, 2026-11-28 03:01 UTC)

One server of the installed package, 8 requests, the kernel's function tracer with stacks on the three ways KFD
evicts a process (reading rule in the header; `kfdtrace-v110/trace.txt`, 329 entries; tracer restored to `nop`).

- **It is the SVM path, called from the server's own system calls.** `svm_range_evict` 158 entries:
  102 from `munmap` (`__x64_sys_munmap -> __split_vma -> vma_adjust_trans_huge -> __split_huge_pmd ->
  mmu_notifier_invalidate_range_start -> svm_range_cpu_invalidate_pagetables`), 16 from `madvise`
  (`do_madvise -> zap_page_range_single`), 40 from the runtime's own `svm_ioctl -> svm_range_set_attr ->
  svm_migrate_to_vram` during the model load. `kgd2kfd_schedule_evict_and_restore_process` (a buffer being moved):
  0 entries. `amdgpu_amdkfd_evict_userptr`: only at process exit.
- **Each one stops every queue of the process** (`kgd2kfd_quiesce_mm`, 17 times) until a kernel worker
  (`svm_range_restore_work`) restarts them: 275 ms once during the load (the 0.7 s per process in the eviction log
  is that, summed over four dies), 34 to 40 ms once per request from the third on, 1 to 4 ms otherwise.
- So: the HIP runtime registers host memory with the driver as shared-virtual-memory ranges (and migrates some of it
  to VRAM); when the program frees or unmaps host memory in such a range, or beside one (the kernel announces the
  whole 2 MB around a split point), the driver evicts all queues because gfx906 cannot retry a GPU page fault.
- **What this does and does not settle.** In steady state the cost is small (under 10 ms per request). It is a
  mechanism by which a program's own memory handling stops all four dies with the host idle, which is what the
  pauses look like; that the pauses are this (a restore that takes seconds or repeats) is NOT shown: none occurred
  under the trace. Next: `kfdtrace-v110.sh`-style tracing plus `evsample.sh` around a run that pauses; and whether an
  allocator that never returns memory (or the runtime not using SVM ranges for pageable copies) removes the
  per-request eviction. Not package work; recorded in the tuning repository's `NEXT-STEPS.md`.

## Pushed, published, dogfooded (2026-11-28 02:57-02:59 UTC)

Every rule of the queue met ((C3) on the rerun), so:

- **Fork** (`push-v110.sh`, `push-v110.out`): `master` = `gfx906-single` = `gfx906-multi` = `c782c317d` (was
  `224b73e24`), `gfx906-candidates` = `797abac2e` (was `5c01f4ffd`), annotated tags `gfx906/v1.1.0/r0/merge`
  (`a3aa1c772`), `/master` (`c782c317d`), `/candidates` (`797abac2e`); the three `wip/*` branches deleted. All
  force-with-lease against the heads read before. PR mxxm-t#25 untouched (open, head `a3aa1c772`, no comment).
- **Package** (`publish-llama-110.sh`, `publish-llama-110.out`): release
  `mx-llama-cpp-1.1.0-gfx906-rocm10.1` on exabit-io/resize-amdgpu-bars (signed flat apt repository, notes
  `/root/mx-llama-deb/release-notes-1.1.0-rocm10.1.final.md`), and the deb added to the rolling channel
  `mx-llama-cpp-gfx906` (8 debs, 1.1.0 newest; channel page updated, previous page kept as
  `/root/local-apt/mx-llama-cpp-gfx906-channel-notes.pre-1.1.0.md`). Both verified over https by the scripts.
- **Dogfood** (`dogfood-v110/progress`): `apt-get update` clean, reinstall from the channel, chain verified
  (InRelease signature -> Packages SHA256 -> deb SHA256 -> origin signature in the deb), channel deb identical to the
  local build, 26 ROCm libraries all from `/opt/rocm/core-10.1`. The box runs
  `1.1.0+git20261127.c782c317d.rocm10.1.gfx906.20261018172427-1` from the channel.

For the record of the arms: `a6` (`9a7177d0a`, `/root/build-master-v110`) has the tree of `e2497e7c0` = published
`master~4` (the port-note rewrite changed messages only).

**Open after this cycle** (also in the tuning repository's `NEXT-STEPS.md`): the pauses (cause); the per-request SVM
eviction (lever: allocator or runtime setting); a build without RCCL wants a wider gate than 16 rows (not built);
17 to 23 slots untimed; which of the two defaults flips short-prompt greedy text (not isolated); rs-modes3 rule 3 and
the KV-replica defect (the fork's, unchanged); the lead's decisions: the rates comment for PR #25
(`pr25-rates-comment.md`, not posted), the staged clock-latch test, whether PR #25 stays open.
