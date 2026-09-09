# NEXT-STEPS measurements, 2026-09-08 (M2, M4, fork knobs, kernel fusions, M3, pairs, batch-1 MMVQ)

Conditions as in the run-through: perf level high, fans max, host RAPL 150 W, the 5 s clock sampler and the SMC log beside every job, watchdog v2. Production build = `/opt/llama.cpp-mxxm-fh` (fork tile table + both MMVQ patches), run with its own `LD_LIBRARY_PATH`. Scripts in `tools/` (copied from `/root/rocm-tests/bench`), raw outputs `qwen38-27b-*-{m2-slots,m4-kvu8x32k,forkknobs,nq-test,m3-mtp,tp2x2,b1-sweep}.md` on the box.

## M2 — production build at 24 and 32 slots, and 12–32 at depth (`m2-slots-prod.sh`)

tp4 `llama-batched-bench -ntg 128`, f16 KV, `-b 2048 -ub 2048 -fa on`; 2K at `-c 69632`, 8K/32K at `-c 1052672`.

| prompt | slots | prefill t/s | decode t/s | per stream | vs 16 slots |
|---:|---:|---:|---:|---:|---:|
| 2048 | 8 | 1140 | 175.1 | 21.9 | −14% |
| 2048 | 12 | 1143 | 197.3 | 16.4 | −3% |
| 2048 | 16 | 1143 | 203.7 | 12.7 | — |
| 2048 | 24 | 1143 | 192.6 | 8.0 | −5% |
| 2048 | 32 | 1143 | **212.7** | 6.6 | **+4.4%** |
| 8192 | 12 | 1101 | 185.9 | 15.5 | −4% |
| 8192 | 16 | 1101 | 192.8 | 12.0 | — |
| 8192 | 24 | 1100 | 183.2 | 7.6 | −5% |
| 8192 | 32 | 1100 | **202.8** | 6.3 | **+5.2%** |
| 32768 | 12 | 959 | 148.7 | 12.4 | −7% |
| 32768 | 16 | 959 | 159.6 | 10.0 | — |
| 32768 | 24 | 959 | 152.5 | 6.4 | −4% |
| 32768 | 32 | 959 | **168.0** | 5.2 | **+5.2%** |

- 32 slots beat 16 by 4–5% at every depth; 24 slots are below 16 everywhere (the 24-wide tile does not pay for the extra rows).
- Against stock at 2K (161.7 / 179.2 at 24 / 32), the fork's gfx906 MMQ tile table lifts the 24/32-slot decode by 19%, less than its 33% on prefill: the decode-shaped tile (32 columns) still has ~10% on the table (NEXT-STEPS S7).
- 16 slots hold at depth: −5% at 8K, −22% at 32K.
- The optimiser's busy-server scenario (≥ 6 tok/s per stream, 32K per slot) moved to `-np 32` (`settings/launch.sh busy`); the team scenario (≥ 12 tok/s) stays at 16.
- Power: the SMC DC total peaked at **1199 W** during the 32-slot 32K prefill at the 200 W caps with the host at 150 W RAPL, 29 W under the 1228 W envelope. Serving at 32 slots on a dedicated node at the full cap is inside the envelope only with the host capped; the hyperconverged nodes run at 125 W anyway.

## M4 — 8 × 32K pooled decode on b10837 (`m4-kvu-8x32k.sh`)

b10837 (upstream, stock kernels), tp4 `llama-batched-bench -npl 8 -npp 32768 -ntg 128 -c 263168`.

| mode | prefill t/s | decode t/s | per stream |
|---|---:|---:|---:|
| private slots | 742 | 128.9 | 16.1 |
| `--kv-unified` pool | 401 | 85.5 | 10.7 |

The pool costs 34% of decode and 46% of prefill at 8 × 32K, the same loss as on b10288 (130 / 86). The upstream change that made the pooled server read a lone prompt at full speed (run-through s.4) did not touch the batched pooled path. `--kv-unified` stays a capacity mode (8 × 256K fits through it); nothing in the guide changes.

## Fork knobs — the fork's custom XGMI allreduce and whole-token graph (`forkknobs-sweep.sh`)

The ML-gfx906 fork tree (mx-llama.cpp b10254, commit 751b611) ships two things the roadmap's S2 was going to build: a peer-write custom allreduce (`ggml-cuda/tp-allreduce.cu`, vLLM-style, one kernel per rank, flags in fine-grained memory) and a whole-token HIP graph per lane (`ggml-backend-meta.cpp`, records the subgraph / allreduce-rank / subgraph sequence through an outer capture). Both are switched by environment variables and neither was on in any measurement so far: the custom AR needs `GGML_ENABLE_CUSTOM_AR=1` and, on gfx906, `HSA_FORCE_FINE_GRAIN_PCIE=1` for the broadcast kernel (without it the code falls back to RCCL when RCCL is present); the token graph is on by default but only records on the custom AR. Production build, tp4.

| variant | env | pp2048 | tg128 | 8 slots 2K | 16 slots 2K |
|---|---|---:|---:|---:|---:|
| baseline (RCCL) | | 1128 | 47.7 | 174.5 | 203.8 |
| custom AR, no fine-grained | `GGML_ENABLE_CUSTOM_AR=1` | 1129 | 47.7 | 174.3 | 204.1 |
| **peer-write AR + token graph** | `+ HSA_FORCE_FINE_GRAIN_PCIE=1` | 1129 | **54.3 (+14%)** | 160.2 (−8%) | 187.8 (−8%) |
| peer-write AR, token graph off | `+ GGML_META_TOKEN_GRAPH=0` | 1129 | 52.4 (+10%) | 155.1 | 182.1 |
| peer-write AR, no size gate | `+ GGML_TP_AR_NO_GATE=1` | **720 (−36%)** | 54.2 | 158.5 | 185.0 |
| fine-grained memory only | `HSA_FORCE_FINE_GRAIN_PCIE=1` | 1129 | 47.8 | 174.8 | 204.3 |
| serial lane dispatch | `GGML_META_PARALLEL_DISPATCH=0` | 1127 | 46.9 | 173.9 | 203.1 |
| baseline again | | 1129 | 47.7 | 174.6 | 204.1 |

- **Single stream +14%** (47.7 → 54.3 tok/s): the peer-write kernel is worth +10% and the whole-token graph another +3.6% on top. Perplexity through the custom path is 5.5969, identical to RCCL. The trace confirms the mechanism: 4 `hipGraphLaunch` per token (one per die) instead of 516, and `k_broadcast_reduce<4>` at 46 µs average per allreduce including its spin-wait.
- **8 and 16 slots lose 8%** because the fork's size gate (`ne < 262144` elements for four ranks) also sends their 8–16-row messages (160–320 KB) through the peer-write kernel, where RCCL is faster. The crossover lies between one and eight rows: the gate sweep below finds it, with a new `GGML_TP_AR_MAX_NE` knob (fusion tree commit 25e1d46).
- Disabling the gate puts prefill through the F32 two-shot kernel and costs a third of it: never.
- Fine-grained memory by itself changes nothing, so the knob is free to set; serial dispatch costs 1.7%.
- This is S2 delivered by the fork: the roadmap's own allreduce kernel is not needed. What remains for S2 is the gate and, possibly, a lower-latency variant of the fork's kernel (46 µs against the 27 µs RCCL kernel it replaces — the gain comes from the graph and the removed host round trips, not from the kernel itself).

## Allreduce size gate (`ar-gate-sweep.sh`, fusion build)

`GGML_TP_AR_MAX_NE` (fusion-tree commit 25e1d46) replaces the fork's fixed gate (262144 elements for four ranks) with an explicit element count; one decode row is 5120 elements. tp4, `llama-batched-bench -npl 1,2,4,8,16` at 2K, decode tok/s; single stream from `llama-bench tg128 -r 3`.

| variant | gate | 1 slot | 2 | 4 | 8 | 16 | tg128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RCCL (base) | — | 48.9 | 77.1 | 122.0 | 175.0 | 203.9 | 49.4 |
| peer-write, fork default | 262144 | 55.8 | 92.2 | 124.9 | 160.1 | 187.4 | |
| peer-write, 1 row | 5121 | 52.1 | 76.4 | 121.1 | 175.3 | 203.9 | **56.6 (+14.5%)** |
| peer-write, 2 rows | 10241 | 44.9 | 92.9 | 122.1 | 175.7 | 203.9 | |
| **peer-write, 4 rows** | **20481** | 54.3 | **91.7 (+19%)** | 123.6 (+1%) | 174.9 | 204.1 | |
| peer-write, 8 rows | 40961 | 54.6 | 93.0 | 125.5 (+3%) | 166.6 (−5%) | 203.8 | |
| RCCL again | — | 48.8 | 77.0 | 122.5 | 174.8 | 203.7 | |

- The crossover is at four rows: two-row messages gain 19–20%, four-row ones 1–3%, eight-row ones lose 5%. The batched bench's one-slot cell is not reliable with the custom path (44.9–55.8 across gates that treat one row identically: the whole-token graph warms up during the first batch), so the single-stream figure is llama-bench's 56.6 against 49.4.
- **Adopted for builds that carry the knob:** `GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481`. One and two streams gain 14–20%, four streams (also the MTP verify batch at draft 3) 1–3%, eight and more are untouched. Perplexity through the custom path is 5.5969.
- On the current production binary (no knob) the fork's default gate applies, which costs 8% at 8–16 slots: do not enable the custom AR there for multi-user profiles; the knob ships with the next production build.

## Kernel fusions, S3 (`nq-test.sh`, `nq-check2.sh`; patches/0001–0003 on the production source)

Three fusions built into `/opt/llama.cpp-mxxm-fh-nq`, each with an off switch: (1) the fused RMS_NORM+MUL writes the Q8_1 copy of its output into the q8_1 cache (128 of the 257 quantize launches: the ones after a norm); (2) the residual ADD is computed inside the norm kernel (129 launches); (3) the delta-net block's q/k L2 norms, beta sigmoid and gate chain are recomputed inside the GDN kernel (the gate chain did not match; 144 launches). Kernels per token per die 1866 → 1466, busy 20.4 → 19.3 ms.

| variant | tp4 pp2048 | tp4 tg128 | one die pp2048 | one die tg128 | greedy = production | PPL 16K |
|---|---:|---:|---:|---:|---|---|
| production | 1128 | 48.2 | 322 | 20.32 | — | 5.5969 |
| fusions, all on | 1052 | 49.5 (+2.8%) | 311 | 20.45 | no (char 556) | 5.6083 |
| all off | 1128 | 48.0 | 328 | 20.62 | yes | 5.5969 |
| q8 off | 1054 | 49.8 | 311 | 20.61 | | |
| add+norm off | 1049 | 48.7 | 311 | 20.62 | | |
| GDN fold off | 1135 | 48.6 | 329 | 20.34 | yes | |
| q8 only | | | | | yes | 5.5969 |
| add+norm only | | | | | yes | 5.5969 |
| GDN fold only | | | | | no (char 556) | 5.6083 |

- **Correctness:** the q8 fusion and the add+norm fusion are bit-exact (identical greedy text, perplexity 5.5969). The GDN fold changes the numerics (5.6083, greedy diverges) — being isolated to the q/k norm or the sigmoid part (`gdn-subfold-check.sh`); it is not shippable until that is understood.
- **Prefill:** the GDN fold cost 7% at tp4 (the strided raw q/k views walked token by token inside the prefill GDN kernel); the fold and the add+norm fusion are now restricted to ≤ 64 rows (commit 463bab8), where their launch saving lives.
- **Decode:** add+norm +1.6% single stream, GDN fold +1.8%, q8 fusion 0 (−0.6%): the trace shows why — the q8-emitting norm kernel takes 10.5 µs where norm + quantize took 6.4 + 3.9. **These small kernels are latency-bound, not launch-bound**: their time is their own dependent chain (block reductions, strided loads, byte stores), and fusing two dependent phases keeps both latencies. The M1-based S3 estimate (+15–20%) assumed the ~4 µs per kernel was dispatch overhead; it is not. 8/16-slot decode is unchanged by all three.
- **What this changes in the roadmap:** fusion is worth the ~3% it measured, not more. The lever for the 5.8 ms of small kernels is *overlap*, not fusion: independent branches (q/k/v matvecs, the norms of the two attention paths) on separate streams. The fork's multi-stream graph optimisation (`GGML_CUDA_GRAPH_OPT`) exists but is gated to single-device processes, so on the split it was never on; `=2` now allows it per lane (commit a1462e0) and is being measured (`graphopt-check.sh`).

## M3 — MTP on the production build (`m3-mtp-prod-real.sh`)

tp4 `llama-server`, greedy, 300 generated tokens, N concurrent distinct wikitext prompts, two waves; the decode-only wave 2 is the number that matters (sum of per-request generation rates while all N slots decode).

| prompt | slots | draft | decode tok/s | vs none | acceptance |
|---:|---:|---:|---:|---:|---:|
| 2K | 4 | 0 | 116.8 | — | |
| 2K | 4 | 1 | 132.8 | +14% | 0.84 |
| 2K | 4 | 2 | 134.0 | +15% | 0.72 |
| 2K | 4 | 3 | 133.0 | +14% | 0.64 |
| 2K | 8 | 0 | 161.1 | — | |
| 2K | 8 | 1 | 157.6 | −2% | 0.86 |
| 32K | 4 | 0 | 100.6 | — | |
| 32K | 4 | 1 | 116.4 | +16% | 0.88 |
| 32K | 4 | 2 | 121.6 | +21% | 0.78 |
| 32K | 4 | 3 | **123.6** | **+23%** | 0.73 |
| 32K | 8 | 0 | 133.0 | — | |
| 32K | 8 | 1 | 140.0 | +5% | 0.89 |

- **The verify-batch rule relaxes from ≤ 8 to ≤ 16 rows.** On stock, 4 slots × draft 2 lost 12% at 32K (the 12-row verify batch fell into the MMQ tile); on the production build the 16-column kernel takes it, and 4 slots × draft 3 is +23% at 32K, +14% at 2K.
- 8 slots × draft 1 (16 rows) is neutral at 2K and +5% at 32K; above 16 rows nothing was measured and the stock loss is expected.
- Schedule for S5 (adaptive drafting): slots × (draft + 1) ≤ 16 on the production build, i.e. draft 3 at 1–4 slots, draft 1 at 8 (worth it only at depth), none above 8.

## Two tensor-split pairs on the production build (`tp2x2-prod-real.sh`, TODO item 8)

Two `llama-server` instances (rocm0+rocm1 and rocm2+rocm3, `-sm tensor`), `dual-server-bench` with 1300-token prompts and 256 generated, aggregate over the wave's wall clock.

| slots per pair | clients | agg gen tok/s | per request | TTFT s | per pair |
|---:|---:|---:|---:|---:|---|
| 8 | 8 | 90.7 | 18.2 | 6.8 | 45.3 / 45.3 |
| 8 | 16 | **100.3** | 10.0 | 8.8 | 50.2 / 50.2 |
| 16 | 16 | 101.7 | 9.7 | 7.9 | 50.8 / 50.8 |
| 16 | 32 | 97.0 | 4.3 | 9.4 | 48.5 / 48.5 |

- Against the production four-die server at 16 slots (run-through s.11: 75.5 / 82.1 tok/s at 8 / 16 clients, TTFT 5.6 s) the pairs lead by **+20% at 8 clients and +22% at 16**, at about 3 s more first-token latency. Against the stock pairs (75.2 / 79.4) the production kernels are worth +21–26%.
- Sixteen slots per pair add nothing (101.7 at 16 clients, 97.0 at 32).
- This is the server-level picture with prompt reading interleaved: one server reads prompts in micro-batches that stall its decoders, two servers overlap one's prefill with the other's decode. The decode-only batched bench still favours tp4 (204 at 16 slots, 213 at 32), so the guide keeps both: tp4 for decode-heavy work (long generations, MTP), the pairs for request traffic with real prompts. The tp4 server at 32 slots is unmeasured at the server level.

- **Fairness caveat (raised 2026-09-08 evening; key `pair_link_sensitivity`, `tools/pair-link.sh`):** on the ring as cabled (Apple A2326 bridge, two Duo modules) each pair runs on one XGMI link (~33 GB/s per direction) while tp4 has the whole ring; a two-isolated-pairs bridge (A2339) would give each pair two links. Measured bound: a pair with *no* direct link at all (the diagonal 0b+1e, two hops or PCIe) loses only 2.5–4.6% of prefill (592 vs 607–620 tok/s) and nothing at decode (38.3 vs 37.8–38.3 single stream, 113.6 vs 113–115 at 8 slots) against the one-link pairs — the pair's allreduce is latency-bound at decode. So a second link could add at most a few percent of prefill and first-token latency to the pairs; their 20% server-level lead is the prefill/decode overlap of two servers, not the link. The A2339 configuration was not tested by choice: models above 64 GB (Flash-Next) need all four dies on one fabric.

## Batch-1 matrix-vector variants, S6 (`b1-sweep.sh`)

New one-column knobs in the fusion tree (`GGML_MMVQ_GCN_ROWS1`, `_NWARPS1`, `_Q8_VDR8_1COL`; upstream GCN runs one row per block with two warps, so every block re-reads the whole activation vector), `llama-bench -p 0 -n 128 -r 3`.

| build | rows / warps / vdr8 at one column | tp4 tg128 | one die tg128 |
|---|---|---:|---:|
| fusion base | 1 / 2 / off | 49.3 | 20.32 |
| r2 | 2 / 2 / off | 49.3 | 21.22 (+4.4%) |
| r4 | 4 / 2 / off | 48.5 (−1.7%) | 21.38 (+5.2%) |
| **v8** | 1 / 2 / **on** | **50.7 (+2.8%)** | **21.77 (+7.1%)** |
| r2v8 | 2 / 2 / on | 50.3 | 21.60 |
| w4 | 1 / 4 / off | 46.9 (−5%) | 20.39 |
| production (no fusions) | 1 / 2 / off | 47.9 | 20.23 |
| fusion base again | | 49.4 | 20.45 |

- The whole-block load per thread (vdr 8: nine aligned dwords and a funnel shift per 32-weight block) is the batch-1 winner on both placements; it lost at batch 8 in the run-through, which is why it is now restricted to one column. Adopted as the default (commit d49570b).
- More rows per block help one die (the activation vector is re-read four times less) but not the split, where the quarter-width matrices already fit their blocks; four warps per row lose.
- The one-column kernel goes from 68% to about 73% of HBM on the split; the rest of S6 (a Q8_0 activation format, K-quant fast paths) stays on the list.

### Delta-net fold, isolated (`gdn-subfold-check.sh`)

With the fold restricted to decode-sized batches: the beta-sigmoid part is bit-exact (greedy identical); the q/k L2 part diverges late in greedy decoding (char 556 of 200 tokens) with perplexity unchanged — the standalone `l2_norm` kernel runs 32 threads per row and sums four squares per thread, the fold summed two per lane over 64 lanes, a different rounding order. The earlier perplexity shift (5.6083) came from the fold running inside the 2048-token prefill GDN steps, now excluded. Commit 797124c reproduces the standalone summation order (fused multiply-adds in the same sequence, then the 32-wide tree), and the final-configuration run checks it with a greedy comparison and a KL divergence at 64-token batches — plain perplexity runs 2048-token batches and never exercises the decode-only folds.

## Multi-stream graph optimisation on the split (`graphopt-check.sh`)

`GGML_CUDA_GRAPH_OPT=1` is the fork's (upstream's) multi-stream execution of independent graph nodes, gated to single-device processes; `=2` (commit a1462e0) allows it on each tensor-split lane.

| variant | tp4 pp2048 | tp4 tg128 | one die pp2048 | one die tg128 |
|---|---:|---:|---:|---:|
| off | 1130 | 49.5 | 328 | 20.48 |
| `GGML_CUDA_GRAPH_OPT=1` | 1132 | 49.5 | 328 | 20.41 |
| `GGML_CUDA_GRAPH_OPT=2` | 1131 | 49.4 | 328 | 20.48 |
| custom AR (adopted settings) | 1131 | **56.5** | 328 | 20.35 |
| custom AR + `GRAPH_OPT=2` | 1130 | 56.4 | 328 | 20.49 |
| off again | 1130 | 49.6 | 328 | 20.57 |

Nothing moves, on one die or on the split (perplexity 5.5969 with it on). The optimiser finds nothing to overlap in this model's decode graph: the small kernels sit on the single dependency chain of each block. Together with the fusion result this closes the small-kernel question for now: their 5.8 ms is intrinsic latency on the critical path, recoverable neither by fusing consecutive kernels nor by the existing stream-level concurrency; only a different execution model (a persistent per-block kernel) would change it, and that is beyond this roadmap. The single-stream levers that did work: the fork's custom allreduce with the four-row gate (+14.5%), the one-column whole-block load (+2.8%), the add+norm and delta-net folds (+3%), and MTP on top.

## The final build and 32 slots at the server level (`server-final.sh`, TODO items 10 and 11)

`llama-server` tp4 on the final build with `gfx906.env`, 1300-token prompts / 256 generated; key `server_final_prod`.

| profile | 4 clients | 8 | 12 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| final, 16 slots | 67.4 | 76.1 | 83.0 | 83.4 (7.4 per user, TTFT 5.8 s) | — |
| final, 32 slots | — | — | — | 85.0 | 80.0 (3.3 per user) |
| 2026-09-07 build, 32 slots | — | — | — | 83.4 | 79.3 |
| final, two tp2 pairs at 8 slots | — | 94.4 | — | 101.9 | — |
| 2026-09-07 build, 16 slots / pairs (earlier rows) | — | 75.5 / 90.7 | — | 82.1 / 100.3 | — |

- **TODO 10:** slots beyond 16 add nothing at the server level. 32 slots give 85.0 at 16 clients and 80.0 at 32 clients against 83.4 on 16 slots; the batched bench's +4–5% (M2) does not survive prompt interleaving, and the per-user rate halves. The busy profile goes back to 16 slots; `-np 32` keeps its decode-only figure for offline generation.
- **TODO 11:** the final build is unchanged where the four-row allreduce gate predicts it (16 and 32 slots, +1–2%); the pairs gain 4% at 8 clients (four clients per pair fall inside the gate) and 1.6% at 16.
- DC total peaked at 1162 W during the server runs with the host at 150 W.

## Host load beside the server and the governor threshold, M7 (`governor-threshold.sh`, TODO item 7)

Production build tp4 `-np 16`, 16 clients on 1300/256 requests, beside one `yes` per hardware thread (56) under stepped host RAPL caps; dies at 200 W caps; the job stops at DC 1180 W. Key `governor_threshold_m7`.

| host | agg gen tok/s | per user | TTFT s | DC max W |
|---|---:|---:|---:|---:|
| idle, 150 W cap | 82.3 | 7.4 | 6.50 | 1087 |
| all-core load, 150 W cap | 72.1 (−12%) | 6.3 | 6.67 | 1147 |

| host cap W (all-core load, 16 clients) | 150 | 175 | 200 |
|---|---:|---:|---:|
| DC max over 40 s, W | 1133 | 1172 | 1206 (stop) |

- ~1.5 W of DC per watt of host cap at 200 W die caps: 175 W is the last host cap inside the 1228 W envelope there; at the 125 W production die caps the dies peak ~300 W lower, which is why the host may stay uncapped on the fleet nodes.
- Governor: trigger 1150 W, cut the host cap by 50 W (about −70 W DC); three 5-s samples before the SMC's ~20 s clamp. No clamp occurred (1730 MHz throughout).
- The host load costs the server 12%: the serving threads compete with the tenants; pin them (`--threads`, cpuset) before the governor matters. The tenant-latency half (Ceph/KVM at 125 W) needs a hyperconverged node.

## Cap sweep with the phases separated (`cap-phases.sh`, TODO item 13)

Production build tp4; per cap, `llama-batched-bench -npl 16` decode at 2K (decode-only) and `llama-bench pp2048 -r 3` (prefill-only); caps set live, 200 W repeated at the end. Key `power_cap_phases`.

| cap W | decode 16 slots tok/s | prefill pp2048 tok/s | vs 200 W (decode / prefill / 16-client server aggregate) | die W | sclk MHz |
|---:|---:|---:|---|---:|---:|
| 200 | 203.4 (204.2 at the end) | 1130.0 | — | 200 | 1689 |
| 170 | 194.1 | 1072.8 | −4.6% / −5.1% / −2.7% | 165 | 1598 |
| 140 | 179.5 | 990.9 | −11.8% / −12.3% / −8.9% | 136 | 1459 |
| 125 | 168.3 | 932.1 | −17.3% / −17.5% / −13.5% | 126 | 1389 |
| 85 | 137.7 | 732.9 | −32.4% / −35.1% / −30% | 88 | 999 |

- The two phases fall together at every cap, and both faster than the server aggregate of the power study: the aggregate's shallower curve is prompt reads and queueing, not the dies. The review's double-count concern is settled the other way round — the aggregate under-penalised decode.
- `optimize.py` now takes the 8+-stream decode factor and every prefill factor from these curves (`f_cap`); the sclk model it used below 150 W over-penalised prefill at the floor (−42% modelled, −35% measured). `results.md` regenerated by `tools/gen-results.py`.

## Tile-table ablation (`ablation-bench.sh`, TODO item 12)

The fork's three MMQ commits applied cumulatively on stock b10288: a = tile-load threads-per-row, b = a + the gfx906 MMQ config table with its dispatch guards, c = b + the Q8_0 partial-k unroll; reference the whole fork b10254. Key `tile_table_ablation`; source `reports/2026-09-08-tile-table-ablation.md`.

| build | Q8_0 tp4 pp2048 | Q8_0 one die | Q6_K tp4 | Q4_K_M tp4 | decode 16 / 24 / 32 slots |
|---|---:|---:|---:|---:|---|
| stock b10288 | 844.6 | 230.3 | 660.0 | 747.1 | 141 / 163 / 181 |
| a: tile load | 850.2 | 231.0 | 661.8 | 748.7 | |
| b: + config table | **1100.7** | **315.4** | 670.9 | 750.9 | |
| c: + k-unroll | 1110.9 | 317.5 | 671.1 | 756.2 | **164 / 197 / 217** |
| whole fork b10254 | 1131.1 | 322.4 | 829.1 | 796.6 | 167 / 193 / 213 |

- The config table is the gain: +30% four-die and +37% one-die Q8_0 prefill by itself (89% of the whole fork's), and all of the 16–32-slot decode gain; the tile-load commit +0.7%, the k-unroll +1%, the rest of the fork's 81-file diff +2%. Single-stream decode untouched.
- On the K-quants the table alone is +1–2%; the fork's +26% (Q6_K) and +7% (Q4_K_M) come from its K-quant kernels. S1's upstream PR is one header and one dispatch guard; the K-quant kernels follow as their own patch.

## Flash-attention counters at head size 256, S4 (`fa-counters.sh`, TODO item 9)

Production build tp4; `rocprofv3` kernel trace with `SQ_INSTS_VALU` (the only one of the four requested counters defined for gfx906 on ROCm 7.14; the gfx906-set pass is queued as `fa-counters-2.sh`), plus instruction counts from the gfx906 code object. Primary source `reports/2026-09-08-fa-counters.md`; key `fa_counters_head256`.

| kernel | share of kernel time | occupancy | VALU issue | loop body | reading |
|---|---:|---|---:|---|---|
| `flash_attn_tile<256,256,16,2>` (prefill, 4 × 32K) | 16.6% (MMQ 58%) | 2 waves/SIMD: 89 VGPRs, 26.5 KB LDS | 45% of peak | 1,842 instr: 512 `v_dot2`, 528 `v_pk`, 208 `ds_read`, 193 `s_waitcnt` | latency-bound at low occupancy; a tile at 3 waves/SIMD needs ≤ 84 VGPRs and ≤ 21.8 KB |
| `flash_attn_tile<256,256,1,2>` (decode at 128K) | 16 × 304 µs = 4.9 ms of the 27.9 ms token (+0.5 ms combine) | 3 waves/SIMD: 80 VGPRs, 20 KB | 17% | 840 instr: 128 `v_dot2`, 132 `v_pk`, 129 `ds_read`, 118 `s_waitcnt` | KV-streaming at ≥ 441 GB/s, and **three passes over the same KV** (GQA 6 packed as `ncols2 = 2`); a six-head packing reads it once, floor 150 µs |

- `v_dot2_f32_f16` is in use in both kernels (the review's correction holds); packed-math work is not the lever.
- S4's order becomes: corrected counters → six-head decode packing (dispatch/template change, up to 9% of the 128K token) → three-wave prefill tile.

## Paired confidence intervals for the production table (`ci-paired.sh`, TODO item 14)

Three builds interleaved per round (stock, 2026-09-07 production, production + `gfx906.env`); `llama-bench -p 2048 -n 128 -r 5` × 3 rounds, `llama-batched-bench` × 2 rounds. Key `paired_ci_prod_table`; source `qwen38-27b-ci-paired.md`.

| cell (mean ± sd across rounds) | stock | 2026-09-07 production | production |
|---|---:|---:|---:|
| tp4 pp2048 | 844.0 ± 0.6 | 1128.0 ± 2.4 | 1127.5 ± 2.3 |
| tp4 tg128 | 46.5 ± 0.5 | 47.7 ± 0.7 | **57.9 ± 0.3** |
| rocm0 pp2048 | 233.8 ± 0.2 | 321.9 ± 0.3 | **331.4 ± 0.1** |
| rocm0 tg128 | 20.16 ± 0.05 | 20.33 ± 0.14 | **21.74 ± 0.06** |
| tp4 decode 8 / 12 / 16 slots | 142.6 ± 10.3 / 122.7 / 152.6 | 174.5 / 197.4 / 203.9 | 174.6 / 197.1 / 202.8 |
| rocm0 decode 4 / 8 | 46.7 ± 0.7 / 52.7 | 53.2 / 69.1 | 53.4 / **70.3** |

- Every gain the guide quotes for the 2026-09-08 build is many times its spread: single stream +21%, one die +7%, one-die prefill +3%, one-die batch 8 +1.6%; tp4 prefill and tp4 8/12/16-slot decode tie within 0.3% (batched sd ≤ 0.3).
- Within one `llama-bench` run the split's single-stream spread is 1–2.7 tok/s (the RCCL path); stock's tp4 batch-8 cell is unstable (135 / 150, the MMVQ eight-column boundary). Rounds, not repeats, are the unit for the split.

## The combined configuration against the 2026-09-07 production build (`final-config.sh`)

Fusion build (`/opt/llama.cpp-mxxm-fh-nq`, fusion tree 797124c = the production source plus `patches/0001–0009`) with `GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481`, all folds on.

| build | tp4 pp2048 | tp4 tg128 | one die pp2048 | one die tg128 |
|---|---:|---:|---:|---:|
| production (2026-09-07) | 1130 | 48.3 | 322 | 20.66 |
| **final** | 1130 | **58.1 (+20%)** | 332 (+3%) | **21.76 (+6.5%)** |
| final, allreduce off | 1129 | 50.6 (+5%) | 332 | 21.73 |
| production again | 1131 | 47.9 | 322 | 20.17 |
| final again | 1130 | 57.8 | 332 | 21.71 |

| build | 2 slots | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| production | 76.6 | 123.0 | 174.7 | 203.9 | 212.7 |
| final | 90.8 (+18.5%) | 126.1 (+2.5%) | 174.9 | 202.7 | 214.6 |

| single user, MTP draft 3 | 2K | 32K |
|---|---:|---:|
| production | 75.3 | 67.9 |
| final | 78.5 (+4%) | 73.9 (+9%) |

- **Numerics:** perplexity 5.5969, the production figure. KL divergence at 64-token batches over 16K tokens against the production logits: mean log-ratio 0.000000, 90th-percentile KLD 0.000015, same top token 100%. The greedy text diverges from the 2026-09-07 binary at character 556 of 200 tokens; on the new build the delta-net fold and the custom allreduce are each greedy-identical to their off states, so the divergence is the one-column whole-block load's summation order — the same class of change as the accepted MMVQ rewrites.
- **Adopted as the production build.** `/opt/llama.cpp-prod` points at it; `settings/launch.sh` defaults to it and `settings/gfx906.env` carries the allreduce settings. The 2026-09-07 binary stays at `/opt/llama.cpp-mxxm-fh` as the reference for these numbers.
