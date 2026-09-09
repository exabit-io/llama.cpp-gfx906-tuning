# Flash-attention counters at head size 256 (BENCHMARKS-TODO 9, NEXT-STEPS S4) — 2026-09-08

Production build (`/opt/llama.cpp-prod`, `settings/gfx906.env`), tp4, Qwen3.8-27B Q8_0 (24 query heads, 4 KV heads, head size 256: one KV head and six query heads per die on the split). `tools/fa-counters.sh`: `rocprofv3 --pmc … --kernel-trace --stats` around (a) `llama-batched-bench -npl 4 -npp 32768 -ntg 1` (prefill, 2048-token micro-batches) and (b) `llama-bench -d 131072 -n 32` (decode at 128K; its depth prefill runs at the bench's default 512-token micro-batch). Die clocks during the passes: 1675 / 1690 MHz mean (sampler), 200 W caps. Static figures from the gfx906 code object of `libggml-hip.so` (`clang-offload-bundler`, `llvm-objdump --mcpu=gfx906`, `llvm-readelf --notes`).

**Only `SQ_INSTS_VALU` of the four requested counters exists for gfx906 on ROCm 7.14** (`SQ_BUSY_CYCLES`, `SQ_WAIT_INST_ANY`, `TCP_TOTAL_CACHE_ACCESSES` are not defined; `rocprofv3-avail` lists `GRBM_GUI_ACTIVE`, `SQ_WAVES`, `SQ_ACTIVE_INST_VALU`, `SQ_INSTS_VMEM_RD`, `SQ_INSTS_LDS`, `SQ_WAIT_INST_LDS`, `SQ_LDS_BANK_CONFLICT`, `TCC_HIT`, `TCC_MISS` and derived `VALUBusy`, `VALUUtilization`, `L2CacheHit`). The corrected pass (`tools/fa-counters-2.sh`, three counter groups, queue 23) fills the VALU-busy, LDS-wait and L2-hit columns; what follows is the kernel-trace and instruction-count evidence, which does not depend on it.

## Kernel time share

| phase | kernel | share | dispatches | mean µs | max µs |
|---|---|---:|---:|---:|---:|
| prefill 4 × 32K | `mul_mat_q<Q8_0, 128>` | 58.0% | 94,208 | 3,380 | 4,994 |
| | `flash_attn_tile<256,256,16,2>` | 16.6% | 4,160 | 21,928 | 45,129 |
| | RCCL | 10.1% | 33,792 | 1,635 | |
| | `gated_delta_net<128>` | 5.8% | 12,672 | 2,503 | |
| decode at 128K (llama-bench) | `mul_mat_q<Q8_0, 128>` (depth prefill) | 37.2% | 327,680 | 1,208 | |
| | `flash_attn_tile<256,256,16,2>` (depth prefill, 512-token µbatch) | 33.9% | 16,384 | 22,046 | 56,853 |
| | RCCL | 13.6% | 147,968 | 975 | |
| | `flash_attn_tile<256,256,1,2>` (the decode tokens) | 0.1% | 2,112 | 304 | 351 |
| | `flash_attn_combine_results<256>` | 0.1% | 18,496 | 33 | 75 |

Attention is 17% of a 32K prompt's kernel time on the split (the matrix multiply is 58%); at 128K depth a decode token spends 16 × 304 µs = 4.9 ms in the tile kernel plus 0.5 ms in the combine kernel, 19% of the 27.9 ms token (35.8 tok/s single stream at 128K).

## The prefill kernel `flash_attn_tile<256,256,16,2>` (16 query columns × 2 heads per block)

| | |
|---|---|
| grid per dispatch (2048-token µbatch) | 128 query tiles × 16 KV splits × 3 head pairs = 6,144 workgroups of 256 threads (4 waves); 24,576 waves per dispatch, 384 per CU |
| registers / LDS | 89 VGPRs (92 allocated), 48 SGPRs, 27,136 B LDS, no scratch, no spills |
| occupancy | **2 waves per SIMD**, capped by both limits: ⌊256 / 92⌋ = 2 and ⌊65,536 / 27,136⌋ = 2 workgroups per CU |
| VALU instructions per dispatch | 1.07 × 10⁹ wave-instructions (SQ_INSTS_VALU, mean over 4,096 full-size dispatches) |
| VALU issue rate | 4.8 × 10¹⁰ /s per die = **45% of the issue peak** (256 SIMDs × 1.69 GHz / 4 cycles per wave64 instruction = 1.08 × 10¹¹ /s) |
| static size | 2,678 instructions; 9 `s_barrier`; 19 global loads, 12 global stores |
| KV loop body | 1,842 instructions: 512 `v_dot2_f32_f16`, 528 `v_pk_*` (packed fp16 FMA for P·V and the scale/max bookkeeping), 288 other VALU, **208 `ds_read`, 193 `s_waitcnt`**, 35 moves, 18 SALU |

Reading: the loop is 1,040 dot/packed instructions per 208 LDS reads and 193 wait-counts, run by only two waves per SIMD; each `s_waitcnt` that finds its LDS or global data not yet there stalls the SIMD with one other wave to cover it. That is a latency-bound loop at low occupancy (the MMVQ diagnosis of the run-through in a different kernel), consistent with 45% VALU issue. The tile table's first target is therefore a shape that admits three waves per SIMD, which needs **≤ 84 VGPRs and ≤ 21.8 KB of LDS** (K/V tile of 256 dims at a shorter KV stride or a narrower query tile); four waves would need ≤ 64 VGPRs and ≤ 16 KB. The v_dot2 path is already in use (the 2026-09-08 review's correction stands); nothing here calls for packed-math work before the occupancy work.

## The decode kernel `flash_attn_tile<256,256,1,2>` (1 query column × 2 heads per block)

| | |
|---|---|
| grid per dispatch at 128K | 1 × 512 KV splits × 3 head pairs = 1,536 workgroups (6,144 waves) |
| registers / LDS | 80 VGPRs, 53 SGPRs, 20,480 B LDS (20,000 declared) → 3 waves per SIMD |
| VALU instructions per dispatch | 5.7 × 10⁶ → 1.9 × 10¹⁰ /s = **17% of the issue peak**: not compute-bound |
| KV bytes per dispatch (one KV head, f16, 131,072 tokens × 256 × 2 × 2 B) | 134 MB → **≥ 441 GB/s** HBM-side over the 304 µs (50% of the 880 GB/s measured read bandwidth) |
| passes over the KV | **three**: GQA ratio 6 with `ncols2 = 2` (`fattn.cu` picks 8 / 4 / 2 by divisibility; 6 is only divisible by 2), so the Z = 3 head-pair blocks each stream the same KV slice; 403 MB of L2-side traffic per dispatch = 1.3 TB/s, above HBM bandwidth, so part of the second and third passes hit the 4 MB L2 (measured by `TCC_HIT`/`TCC_MISS` in the corrected pass) |
| static size | 1,361 instructions; KV loop 840: 128 `v_dot2`, 132 `v_pk_*`, 129 `ds_read`, 34 `ds_write`, 118 `s_waitcnt`; 18 barriers, 35 global loads |

Reading: at depth the decode kernel is a KV-streaming kernel that reads the cache up to three times because the head packing does not fit the model's GQA ratio of 6. The instantiations present in the binary for head 256 at one column are `ncols2` = 2, 4, 8; a gqa-6 packing (all six query heads per block: `ncols2 = 8` with two idle lanes, or a 3-head variant) reads each KV byte once and bounds the kernel at the 134 MB / 880 GB/s = 150 µs floor instead of 304 µs — up to 2.5 ms of the 27.9 ms token at 128K (9%), more at 256K. That is the decode half of S4; it is a dispatch-and-template change, not a new kernel.

## What S4 becomes

1. Corrected counters (queue 23): VALU busy and L2 hit rate on both kernels settle the two readings above.
2. Decode: a six-head packing for the `ncols1 = 1` tile kernel (read the KV once per token per die); acceptance = identical logits and the 128K/256K decode rows of the depth ladder.
3. Prefill: a gfx906 tile shape for the 16-column kernel at three waves per SIMD; acceptance = perplexity unchanged, `test-backend-ops` FA cases, the 32K–256K prefill rows.

Files: `/root/rocm-tests/bench/trace-fa/{prefill,decode}/*.csv` (1.2 GB and 2 GB; kernel trace + counter collection), `qwen38-27b-fa-counters*.md/.err`, disassembly in the session scratchpad (`b63.s`, regenerable from `/opt/llama.cpp-prod/lib/libggml-hip.so`).

## Second pass: the gfx906 counter set (queue 23, `tools/fa-counters-2.sh`, key `fa_counters_head256.pass2_gfx906_counters`)

Three counter groups, production build tp4, prefill 4 × 16K (2048-token micro-batches); the second workload (4 × 32K with 16 generated tokens) is dominated by its own prefill, so the one-column decode kernel at depth did not reach the top 12 — its counters need a `-ntg 128` pass (follow-up inside S4).

| kernel (prefill 4 × 32K pass) | time | µs/dispatch | VALU busy | VMEM loads / dispatch | LDS instr / dispatch | LDS wait ÷ GUI_ACTIVE | bank conflicts | L2 hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mul_mat_q<Q8_0,128>` | 57.9% | 3,381 | 61% | 2.3 M | 42.9 M | 2.40 | 6.15 M | 79% |
| `flash_attn_tile<256,256,16,2>` | 16.6% | 21,963 | **47%** | 7.8 M | **185 M** | **7.65** | 0 | **95%** |
| RCCL | 9.6% | 1,578 | 1% | | | | | |
| `gated_delta_net<128>` | 5.8% | 2,043 | 58% | 17.4 M | 29.9 M | 10.7 | 0 | 96% |

Reading: the first pass's 45% VALU estimate is measured at 46–47%. The tile kernel issues 24 LDS instructions per global load and has about 1.9 waves per shader engine waiting on LDS at any moment, with no bank conflicts and a 95% L2 hit rate on the K/V tiles: an LDS-instruction and LDS-latency bound loop at two waves per SIMD. The three-wave tile shape is the first move; fewer LDS instructions per dot (wider LDS reads, K/V held in registers longer) is the second. Side findings: the MMQ Q8_0 × 128 kernel runs 61% VALU with 6 M LDS bank conflicts per dispatch (S7 material); the GDN kernel is the heaviest LDS waiter (10.7).
