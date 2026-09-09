# Survey of the gfx906 llama.cpp forks, upstream PRs and tuning guides — what our fork should take (2026-09-09)

The list surveyed is `gfx906-llamacpp-forks-and-tuning-urls.md` (compiled the same day). Every fork with kernel work was fetched into a scratch clone that shares objects with `/root/exabit-llama.cpp`, so each one could be diffed against its own merge base with upstream `master`; the non-fork repositories (engines, wikis, overlays, patch collections) were shallow-cloned and read. Upstream PR state was pulled from the GitHub API. The goal was one question per item: **does it hold something measured that our `gfx906` branch (upstream master 2026-09-08 + mx-llama.cpp b10912 + the Exabit series) does not have, and would it pay on four Vega 20 dies serving Qwen3.8-27B Q8_0 at 32K context and above?**

Two ground rules were applied throughout. First, the user's rule from the same day: nothing below 16K context matters and 32K is the realistic floor, so every candidate is judged on what it does at depth and with 8–16 slots, not on pp512/tg128 at 2K. Second, the guide's rule: a claim counts only if the source measured it; ideas without numbers are listed as ideas.

## Summary

Nine things are worth taking, in this order. The first two are hours of work and one of them is a correctness bug in our branch.

| # | Take from | What | Measured by the source | Expected here (32K+, tp4) | Effort |
|---|---|---|---|---|---|
| 1 | alex4300 | **gfx906 row for the head-256 attention tile table** (five lines in `fattn-tile.cuh`); ours is still the RDNA2-inherited row | Qwen3.8-27B geometry, 16K f16, µs per layer: n=1 207→191, n=2 267→251, n=4 283→261, **n=8 451→350**; prefill batch 512 23.8→21.6 ms | attention is 19% of a 128K decode token and 17% of a 32K prompt: about +2–4% decode at depth for 8–16 slots, −9% on the attention share of prefill | hours + A/B |
| 2 | alex4300 (draft PR for mx-llama.cpp) | **MTP draft graph must build `out_ids` whenever `n_outputs > 0`** — with `-np 2` and backend sampling, slot 0's draft acceptance collapses to 0.03–0.10 | acceptance slot 0 / slot 1: master 0.045 / 0.405, fixed 0.405 / 0.546 | **our `gfx906` branch carries the bug** (`src/models/qwen35.cpp` line 540 skips `out_ids` when `n_outputs == n_tokens`); production b10254 predates it. Every multi-slot MTP profile on the branch is wrong until this lands | hours |
| 3 | upstream PR #27210 (stew675) + alex4300 | **adaptive MTP draft depth** (`--spec-type draft-mtp-adaptive`), plus a separate cap for n-gram drafters (`--spec-draft-n-max-ngram`) and per-request `speculative.n_max` | on Qwen3.8-27B Q8_0, tensor split, 2 GPUs: reasoning 1.71×, prose 1.88×, **code 2.60× (fixed-3: 2.31×)**, recall 4.9×; milpster at 120K depth: +9% over fixed depth, +89% on fresh KV vs their old setting | the MTP gate our branch fails is a verify-cost problem; an adaptive depth is the lever that lowers it when acceptance falls at depth. Also the ngram+MTP combination for file-editing traffic (alex: 60→143 tok/s on Ornith) | a day + gate runs at 32K/128K |
| 4 | sixvolts furnace | **DPP / `ds_swizzle` warp reductions on GCN** (`common.cuh`, ~130 lines): `__shfl_xor` lowers to `ds_bpermute` LDS round trips on Vega; DPP does it in VALU | MI50: 31B dense Q4_K tg128 21.93→22.97 (+4.7%), 0.8B +2.9%, prefill unchanged; `test-backend-ops` clean. Upstream tried it (#26466) and parked it behind #27841 | +3–5% decode backend-wide (every norm/softmax/dot epilogue); the fusions in our series reduce more per token than any fork, so the per-reduction saving compounds. Reduction order changes → perplexity check, not bit-exact | a day |
| 5 | upstream PR #27841 (thelittlefireman) | **the S1 tile-table upstreaming has a live vehicle**: this PR's Q8_0 rows are the same finding as Marko's table (512 threads, I=128, J≤128) | Johannes measured on MI50: Q8_0 +46% average, Q2_K +95–395%, but **IQ types −15–21% at ubatch ≥ 64** → CHANGES_REQUESTED, stuck since 2026-09-06 | contribute our Q8_0-only measurements (`patches/upstream-S1/`, +29%/+34% prefill, 32 rows +15%, parity elsewhere) to that PR and propose landing the Q8_0 row first, which is what the maintainer asked for ("just commit the table"). This dissolves the authorship question: the table goes upstream under the PR author's name with our numbers, Marko's commit stays on our branch as is | an afternoon of writing |
| 6 | sixvolts furnace | `mmq_y = 64` on gfx906 (occupancy) | ~2% prefill on 4× MI50 | a row change in the tile table; +1–2% prefill | hours |
| 7 | mx-llama.cpp | three new commits since b10912 (Q5_1 repack, recurrent snapshot ring for speculative rollback, opt-in PLE prefault) | — | routine merge; the snapshot ring matters for MTP on the delta-net layers | routine |
| 8 | upstream | `GGML_FA_ALL_QUANTS` replaced by `GGML_FA_QUANTS` (5a4d0feca, 2026-09-09): a list of compiled K/V combinations with a runtime fallback | — | the 4-bit V-cache capacity build must move to the new flag at the next merge | routine |
| 9 | alex4300 | **wide GEMV geometry for 1–8 columns** (activations for all columns staged once in LDS, 256 threads, 16 threads per row, 64 blocks per chunk): the Q4_0 kernel reaches 530 GB/s at n=1 | Q4_0 4096×14336, µs: n=1 77→54, n=4 100→60, n=8 181→104; Q5_K n=8 884→135; Q6_K 341→162 | our Q8_0 fast path already took +31–37% at 8 columns and we measured "LDS-staged activations" as a loser (−16 to −52%), but not this geometry, and their n=1 gain came from the same structure. A port for Q8_0 is candidate 3 for the one-token kernel (S1b/S6) if candidate 2 falls short | 2–3 days |

Also confirmed by the survey, no code needed: keep **f16 KV for speed** at depth (alex4300: 7–10% per step at 23K; q8_0 KV pays a full-cache f16 conversion every step above one column, 3.7 ms at 23K, 8 ms at 46K, which is the mechanism behind our 15–36% loss); **six-head GQA packing in the attention tile is withdrawn** as an S4 item (alex4300 built it: slower, 352 vs 305 µs per layer, 947 vs 408 at n=3, because the kernel is instruction-bound at 1.9 TFLOP/s, not bandwidth-bound); and the **native q8_0 attention tile** milpster wrote loses 27% against the convert path at 120K, so it is a capacity feature only.

Not worth taking, with reasons, in section 4. The revised order of work is in section 6.

## 1. The forks with kernel work

### iacopPBK/llama.cpp-gfx906 — superseded, one line worth checking
The February 2026 release (`gfx906-2602`, upstream b7924) is the origin of the community's gfx906 kernels: DPP reductions, a Q8 flash-attention kernel with instances for head sizes up to 256, warp-cooperative MMVQ for Q4_0/Q4_1/Q8_0, a fused norm→Q8 kernel, a Q8 cross-op cache, software-pipelined MMQ loads, a `__sincosf` RoPE fusion, a custom SGEMM. The author (DENEB1312, 14 commits in our fork base) moved to mx-llama.cpp and contributed the Q8_0 repack there; the README now points at mx-llama.cpp. What we already have in another form: the Q8 cache (our patch 0001 and the fork's `GGML_CUDA_Q8_1_CACHE`), the fused norm, the MMVQ fast path (ours is measured faster than the warp-cooperative form: alex4300 tried the iacop Q8_0 kernel as their experiment patch 06 and did not adopt it), the MMQ config. Not in our tree: the DPP reductions (taken via furnace, item 4, which is the cleaner port), the Q8 FA kernel (a q8_0-KV path; we run f16 KV for speed), the RoPE `__sincosf` fusion (trivial, unmeasured, RoPE is not in our top kernels). **Verdict: nothing to port directly.**

### milpster/gfx906-llama-cpp — the most current fork; its findings matter more than its code
Tracks upstream master (synced 2026-09-03), 164 commits ahead, run by one person on 2× Radeon VII + an RTX 3080 under Vulkan, `-sm layer`, DFlash2 drafter, 250K context on 40 GB. Its journal (`journal/`, 3,560 lines) and `bench/FINDINGS.md` are the most careful measurement record in the list. What it holds:
- **wave64 MMQ tables** (`mmq-config-vega.cuh`: I=128 at 256 threads for everything, no J>64) — a different answer from Marko's (I=128, 512 threads, J≤128 for Q8_0) and from PR #27841; theirs is tuned on Q6_K. For Q8_0 ours is measured on our shapes; no action.
- **native q8_0 tile FA kernel and a mixed f16-K/q8_0-V kernel** with an AUTO/convert/native selector: measured E61, native 9.2 tok/s vs convert 12.6 at 120K → default OFF, kept as the compact-buffer path. Confirms our KV finding. Not taken.
- **software-pipelined k-steps in the FA tile (QPIPE)**: rejected in their own microbench (E101); **3-blocks-per-CU occupancy row (OCC3)**: identical (E103). Two S4 ideas we do not need to try.
- **frugal compute buffers** (no_alloc reserve, 276 MiB per device against stock's 1,302): buys them 250K on 40 GB; stock's deep TG was 28% faster and they suspect the reuse dependencies. Our dies have 24 GiB free beside the weights; not our constraint.
- **adaptive MTP (PR #27210) integration notes**: +89% TG on fresh KV, +9% at 120K, and a root-caused PP regression in the PR's delta-net conv-state hunk (`t_min` loop bound) that they revert. Take the note with item 3.
- their table of upstream PRs tested and rejected on gfx906 (#21698 q8 loader +1% = noise, #23685 packed Q8_1 MMVQ no gain, #25635 FA XOR swizzle −0.5 to −5.8%, #28313 top-k ≤ 0.4 tok/s, #24546 MoE tiles no headroom for them) saves us the same experiments.
- **MMQ headroom**: their `bench-dot4` puts production MMQ at 41% of the measured `v_dot4` issue rate on Radeon VII; prefill at batch 16384 is compute-bound. That is the ceiling S7 works against.
**Verdict: read, cite, take nothing but the adaptive-MTP note.**

### alex4300/llama.cpp-gfx906-opt — the closest to our problem
One MI50 32 GB, ROCm 7.1, Qwen3.8-27B (Q4_0 for speed on one 32 GB card), upstream master 2026-09-03 + Unsloth's qwen4exp branch, pushed the day of the survey. Every change has a written prediction, the measurement and the raw numbers (German, `docs/gfx906/`, 47 result files). Their headline is single-die decode 26.0→30.0 tok/s and MTP draft-3 37.6→53.1 on Q4_0. What transfers:
- **the head-256 tile row** (item 1): the only swept configuration for Qwen3.8's attention geometry (GQA 6, head 256) anyone has published; ours is the RDNA2 row. Decode wants small tiles and high occupancy, prefill 512 threads and large tiles. Also the finding that the 6-head packing loses (item withdrawn).
- **the MTP `out_ids` bug** (item 2), found with two concurrent agents on `-np 2` and bisected to the b10589 merge; they sent it to mx-llama.cpp as a PR draft on 2026-09-07. It has not landed there.
- **the separate n-gram cap** and **per-request draft length** (item 3): MTP flat and n-gram deep is the optimum for file-editing traffic, and the two drafters need different caps.
- **the wide GEMV** (item 9).
- their **MMQ profile for GCN5** (`mmq-config-gcn5.cuh`, third table in the field; Q4_0-centred).
- their assessment of mx-llama.cpp: on one card their kernels lead the fork by 19% prefill / 13% decode (the fork "has little kernel-side for gfx906: 31 lines in mmvq.cu, nothing in the attention tile"); on their PCIe box the fork's allreduce cannot run (no peer access, 16 GB BAR0). Our XGMI ring is exactly what makes the fork's tensor parallel worth having; the kernel gap they describe is what our series and this survey fill.
- transferable hardware facts (`erkenntnisse-gfx906.md`): full HBM bandwidth from 2 waves/SIMD; odd block strides cost nothing; LDS allocation granularity is sharp (4 bytes more cost 4%); wide loads (`dwordx4`) are worth +26% at 4 waves and +88% at 2 waves per SIMD, which is where MMVQ runs; `__vcmpne4`/`__vsub4` in `vendors/hip.h` are byte loops (25 instructions where 5 do); IQ codebooks in LDS are not worth it; `test-backend-ops` cannot catch buffer-lifetime bugs across HIP-graph replays (run the same request twice).
**Verdict: take items 1, 2, 3 (partly), 9; their tools directory has a kernel-time profiler and a step profiler worth borrowing for M6.**

### mxxm-t/mx-llama.cpp — our base
Three commits since the b10912 tag we merged (2026-09-08/09): Q5_1 on the repack path, a recurrent-state snapshot ring for speculative rollback, opt-in prefault of the PLE table. Routine merge (item 7). The `out_ids` fix (item 2) is not in it.

### arte-fact/llamacpp-gfx-906-turbo, moriyasujapan/…-gemma4, THEman6989 wrapper, Wizard815/mx-llama.cpp-Rocm10 — the TurboQuant family
TurboQuant `turbo3` KV compression (3.5 bits per value, Walsh–Hadamard rotation + Lloyd-Max codebook, 4.6× against f16) on top of the iacop kernels; the arte-fact README measures 300K context on 4× MI50 for an 80B MoE at 37 tok/s against 90K with f16 at 57 tok/s (−18% at equal context). Wizard815's fork is the interesting one: it is **mx-llama.cpp + TurboQuant + furnace's K-quant repack + PR #27841's table + a Q5_K J=64 override**, on a ROCm 10 build — i.e. someone already did the merge of three of this survey's items onto our base (with a fixed type-confusion between the two repack schemes), untested beyond two GPUs.
For the 27B the KV cache is small (16 of 64 blocks are attention; 16 KiB per token on the split): 16 slots × 96K f16 fit beside the weights today. TurboQuant only pays above ~100K per slot at 16 slots, and its quality on our models is unmeasured. **Verdict: capacity feature, not now; when needed, take it from the Wizard815 tree rather than the original.** Stevio2d's `tq3_0`-K + f16-V variant (K only, 39% KV saving, −2 to −3% decode) is the same idea with less risk and the same low priority.

### sixvolts/llamacpp-gfx906-furnace — the DPP reductions and the K-quant repack
21 reviewable commits on upstream b9587 (June 2026), each with an A/B in the message, ported from the author's `reinstinct` engine. Items 4 and 6 come from here. The three-plane **K-quant repack** (Q3_K–Q6_K, +18–24% dense decode on Q4_K, +12% MoE prefill, "~58% → ~89% of HBM" on the matvec) does not apply to Q8_0 — their own Q8_0 repack reads −3% decode against canonical, which matches what we found on the fork's Q8_0 repack — but it is the right thing for **Qwen3.8-Flash-Next**, whose experts are Q4_K in the UD-Q4_K_XL quant. Their in-kernel MoE expert routing (MoE decode parity, was −7%) belongs with it. The TurboPrefill intra-prompt pipelining (+49% prefill on 4× MI50) is for the no-P2P layer split and does not apply to our tensor split. The env-gated per-op GFXPROF profiler is a nice tool. **Verdict: items 4 and 6 now; the K-quant repack when Flash-Next serving is the workload (from Wizard815's already-merged tree).**

### eslowney/llama.cpp-gfx906 — head size 128 only
A hand-written f16 tile FA kernel for D=128 (64-thread wavefront, register blocking, `v_dot2_f32_f16`, `ds_swizzle`), September 2025. alex4300 looked at porting it to D=256 and found its LDS need at 256 (78 KB) exceeds the CU. Its `gfx906-wave-primitives.cuh` is a readable reference for the DPP patterns. **Verdict: reference only.**

### Luna-AI-Infra/Mi50-Qwen38-27B — a different regime
An isolated HIP overlay for Qwen3.8-27B Q3_K_S on a 16 GB MI50 under ROCm 5.7, N=1 only: 13.5→28.8 tok/s by a Q3 storage rearrangement and "exact" Q5/Q6/Q8 GEMV kernels that keep the stock two-wave reduction order. The reports (Chinese) are careful and reach our conclusions from the other side: the matvec is bound by instruction count per dot product, not by HBM; MTP on gfx906 gains nothing on their Q3 path because the verify columns cost. Nothing for Q8_0 beyond our fast path. **Verdict: nothing to take; cite for the cross-GPU comparison table.**

### Kausik-A/gemma4-mi50-optimizations
Shares the Q8_1 activation quantisation across Q/K/V and fuses their row blocks: +1.7% on one MI50. Our fused norm emits the Q8_1 copy once (patch 0001) and the fork has the Q8_1 cache. **Already covered.**

### The rest of section 1
seekkii (iacop's 2602 kernels on a May-2026 base), FreesoSaiFared (an attempt to re-derive the iacop patches as prompts), 362132718 (furnace + upstream #20819/#20822 slot-state persistence — a server ops feature, note for fleet hot-swap), moriyasujapan multibackend (build recipe), renlililoli (Hygon Z100 build fix), the SOLVE_TRI fix forks (the crash is fixed upstream and in our tree), bowmanjd (Nix flake), DByte308 (below, under speculative decoding), mirrors. **Nothing to take.**

## 2. Upstream PRs

| PR | State | Relevance |
|---|---|---|
| #27841 GCN MMQ config (thelittlefireman) | open, CHANGES_REQUESTED 2026-09-04; Johannes's MI50 numbers 2026-09-06: Q8_0 +46% avg, K-quants +6–35%, Q2_K +95–395%, but IQ1_S/IQ2_S/IQ3_*/IQ4_XS **0.79–0.86 at ubatch ≥ 64** | item 5: the vehicle for S1. Its Q8_0 rows are our table. Offer the Q8_0-only split |
| #27210 adaptive MTP depth (stew675) | open, mergeable, review required, updated 2026-09-08; measured on Qwen3.8-27B Q8_0 | item 3 |
| #26466 DPP reductions (thelittlefireman, code from maximumbusdatatype) | closed 2026-08-28 "waiting for #27841" | item 4 via furnace's version |
| #20831 dynamic MMVQ nwarps for narrow matrices | open; RDNA3/RDNA4 only in the code | not for GCN; our MoE path is the fork's |
| #21698 q8_0 load-tiles pipeline (iacopPBK) | open; milpster measured +1% = noise under a Vega table | skip |
| #22466 async pinned upload for `-sm tensor` load (mxxm-t) | open, WIP; the fork already carries its loader | load-time only; nothing to do |
| #24554 TP for 4–10 GPUs | open; a stepfun/laguna model fix | not our models |
| #16000 / #14969 NUMA mirror | open/closed | single socket, no |
| #19434 quant-bench tool | open | a kernel microbench; our `test-backend-ops perf` covers it |
| #20819 / #20822 slot save / auto-restore | open | server ops; revisit for the fleet |
| #24546 MoE N-tiles from typical expert width | merged then **reverted** (e71b80510) | alex4300 called it the one real candidate for MoE on gfx906; watch it for Flash-Next |
| #15884, #15927, #15769, #15982, #16492 (FA on AMD), #11831, #11519, #15802, #24588, #22933, #19378, #23792, #22673, #23398, #16653, #20793 | merged | all in our base |

New on master since our merge (12 commits): only `GGML_FA_QUANTS` (item 8) touches us.

## 3. Tuning guides and knowledge — what transfers

- **skyne98 wiki studies** (Feb 2026, measured on MI50): KV layout `[head][seq][dim]` reads 1.76 TB/s in dot-style decode traversal against 0.37 for `[head][dim][seq]` (upstream's layout is the right one); DPP row shifts ~2× the exchange rate of LDS; `ds_bpermute` beats LDS exchange; LDS `b128` accesses 9.5–11.2 TB/s against 1.9–3.9 for `b32`; `dwordx4` global loads +7%; no `s_clause`/`s_delay_alu` on gfx906, so latency hiding is ILP and `s_waitcnt` placement. All consistent with ISA-NOTES.
- **arkprojects perf-tuning** (mixa3607): PowerPlay-table edits via `upp` — memory 1000→1150 MHz and GPU 1725→1850 MHz gave Gemma4-31B Q8_0 pp2048 430.1→457.5 tok/s (+6%) and tg 32.4→34.1; a TDC limit 350→150 A cut the hotspot by ~10 °C at no measured cost. A hardware knob for S9 on Apple's Vega II MPX cards, inside a 1228 W chassis envelope that already clamps; the user's decision, not a fork item.
- **fankserver** (MI50, MoE): break-even acceptance for MTP on a sparse MoE is ~41–42% because the verify pass activates the union of experts (1.66× a decode for draft 3); MTP holds decode across context better than DFlash (+58% at 65K on Qwen3.6-35B-A3B); the MTP head costs 14% prefill. Relevant to Flash-Next (512 experts): expect a higher break-even there than on the dense 27B.
- **DByte308/ornith-mtp-gfx906**: **distilling the MTP head against the target model's own outputs at long context** lifted next-next-token accuracy 80→90% and live draft acceptance at 32K from 62% to 74% (+11% throughput), all of the gain at long context. The pipeline (extract the nextn block from the GGUF, capture hidden states, train two epochs on one Vega 20, patch the GGUF in place) is model-side and transfers to Qwen3.8-27B. Also: `--spec-draft-p-min` raises acceptance but not throughput on gfx906; a first request with `n_predict: 1` on a fresh server breaks the next drafted request (a real bug to avoid in benchmark harnesses).
- **larkinwc/mi50grad** (custom engine, 4× MI50 over PCIe): the decode-throughput idea we do not have is the **deferred attention allreduce** (halve the allreduces per token by letting the FFN norm read the partial attention sum; cosine ≥ 0.99, i.e. an approximation, ~10–20% for them where the allreduce is 79 µs over PCIe). Ours is 27–46 µs over XGMI and 16% of the token; the approximation changes numerics and fails our exactness rule. Their fused GEMV+allreduce+RMSNorm (+3.8%) is the S2 fusion idea. Their P2P allreduce kernel study: memory-latency-bound, thread-count tuning did nothing.
- **sixvolts/reinstinct** docs: the port plan that became furnace, and a two-tier KV ("SuperQuant", int8 warm / turbo3 cold) that costs 29–35% decode — a capacity feature by their own account.
- **nick413-bit/gfx906-fa-vllm**: the vLLM attention backend for long prompts (direct-paged FA, +20–40% at 130K); its roadmap names the FlashDecoding++ unified-max softmax as the next algorithmic step for `v_dot2_f32_f16` hardware — an S4 idea with a paper behind it.
- **localaiservers**: a full-BAR P2P amdgpu patch (we have the resizable-BAR driver), vLLM profiles for the 27B (TP8 F16, 70 tok/s), QC methodology documents.
- The arkprojects llama.cpp benchmark pages (ROCm 6.3.3 vs 7.2.3, graphs on/off, 1/2/4 GPUs, 0/16K/32K) did not render through the fetcher; the perf-tuning page did.

## 4. Not taken, and why

| Item | Reason |
|---|---|
| iacop / seekkii / THEman Q8 flash-attention kernel | a q8_0-KV path; f16 KV is faster at depth on this chip (alex4300, milpster, ours) |
| milpster native q8_0 tile, QPIPE, OCC3 | their own measurements: −27%, rejected, identical |
| Six-head GQA packing in the tile (was S4's first item) | measured slower by alex4300; the kernel is instruction-bound |
| mi50grad deferred allreduce | approximation; fails the exactness gate |
| TurboQuant / tq3_0 / SuperQuant KV | capacity features at −18–35% decode; the 27B's KV is small; revisit above ~100K per slot × 16 slots |
| furnace TurboPrefill, milpster pipeline/cost split, draft-head mirror | layer-split and Vulkan-hybrid machinery; we run tensor split over XGMI |
| Q3/Q4-specific kernels (Luna, alex4300's Q4_0 GEMV as such) | we serve Q8_0; the geometry is item 9 |
| PR #20831 dynamic nwarps, #21698, #23685, #25635, #28313 | not GCN, or measured as noise by milpster/alex4300 |
| `__vsub4`/`__vcmpne4` SWAR (#27962) | −20% instructions, +0.8% time on gfx906 (alex4300); the loader is bound by load/wait distance |
| `-funsafe-math-optimizations` | no numeric or speed effect on current code (our ppl bisect) |
| HBM/GPU overclock via `upp` | hardware, +5–6%, inside a chassis that already clamps at 1228 W; user's call |

## 5. What the 32K rule changes in our own numbers

Most of the guide's headline cells are at 2K depth: the batched staircase (slots 1–32 at 2K, with 8K and 32K only for M2), the S1b a3 table (12–32 rows measured at 2K only), the paired-CI production table, the width sweeps of the review follow-up, and the server-level gates (1300-token prompts, 256 generated — a 2K-class request). The 32K and 128K cells exist for one, four and eight slots and for the MTP gate. Consequences:

- The **promotion gates** move to 32K: batched-bench at depth 32768 for 8/12/16 slots, the server client with 32K prompts (and a 128K row), the MTP gate as it is (already 32K). `tools/s1b-test.sh` and `tools/server-bench.py` need a depth parameter; `data/benchmarks.json` gets `*_32k` keys and the optimiser's default workload becomes 32K per slot.
- The **S1b a3 result** (+15–20% at 17–32 rows) is unproven at depth; at 32K the attention share grows and the MMVQ share shrinks, so expect less. Re-measure before promotion.
- **Item 1** (the attention tile row) is worth more at 32K than the 2K cells would show, which is why it is first.
- The **capacity modes** (q8_0 KV, 4-bit V) were judged on 8 × 192K/256K; with f16 KV fitting 16 × 96K they matter only for the long tail.

## 6. Revised order of work

1. **Item 2** (MTP `out_ids` fix) — before any multi-slot MTP measurement on the branch.
2. **Item 1** (head-256 tile row) — A/B at 32K on tp4, 1/8/16 slots, and pp2048/32K prefill; keep if attention time falls as alex4300 measured. Then the same sweep for our GQA-6 geometry on four dies (the row is tuned on one card).
3. **Candidate 2** for S1b (one-column whole-block load on the two-plane layout) as planned; **item 9** is candidate 3 if it falls short.
4. **Item 3** (adaptive MTP + ngram cap + per-request depth) — apply PR #27210 with milpster's `t_min` revert, measure the MTP gate at 32K and 128K, code and prose and reasoning.
5. **Item 4** (DPP reductions) and **item 6** (`mmq_y` 64) — one build, one A/B, a perplexity check.
6. Re-run the three promotion gates at 32K (section 5) and promote.
7. **Item 5** — the #27841 comment with our Q8_0 measurements; close S1.
8. S4 next: a tile-parameter sweep on four dies (the row is one card's), then the FlashDecoding++ softmax; six-head packing is off the list.
9. Flash-Next serving: furnace's K-quant repack and MoE routing from the Wizard815 tree, PR #24546's tile choice if it returns.
10. Model-side, when the 27B's MTP acceptance at 32K+ is the limit: the DByte308 head distillation.

## Sources fetched

Forks diffed against their merge base (scratch clone, `refs/forks/*`): iacopPBK (gfx906-2602), milpster (master 2026-09-06), arte-fact, sixvolts furnace (gfx906-perf), eslowney, alex4300 (gfx906, 2026-09-08), seekkii, stevio2d (tq3_0-mi50-slim-pr), THEman6989 turbo-mtp, 362132718 slot-save, moriyasujapan ×2, Wizard815 rocm10, renlililoli, Kausik-A, FreesoSaiFared, bowmanjd, Luskendeilder, mx-llama.cpp master (6ff101bae). Read: Luna-AI-Infra, stevio2d/GFX906-MI50-optimisations, THEman6989 wrapper, DByte308, larkinwc/mi50grad, sixvolts/reinstinct (+ GaryJS3), skyne98/wiki-gfx906, fankserver, joe2gaan/localaiservers, phantomic12 infohub, nick413-bit. Upstream PRs: the 37 numbers in the list, states from the GitHub API on 2026-09-09. Web: arkprojects perf-tuning.
