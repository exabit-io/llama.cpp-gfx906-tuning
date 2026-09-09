# BENCHMARKS-TODO run-through, 2026-09-07

Same box, same build (b10288 / 360e134, ROCm 7.14, kernel 7.0.0-30), same model (Qwen3.8-27B Q8_0). Conditions for every GPU run: `rocm-smi --setperflevel high`, T2 fans at full speed, CPU governor `performance`, 5 s clock/temperature/power sampler beside each run, XGMI probe run before starting (all eight link directions at 33 GB/s, ring 257 GB/s both ways). Raw outputs live in `/root/rocm-tests/bench/qwen38-27b-q8_0-{knobs,tp2x2,mixed2,mtpdepth,mmvq16,cublas,faq,mmvqprof}*` and `qwen38-27b-quant-quality*`; the driver scripts are next to them (`knobs-sweep.sh`, `tp2x2-sweep.sh`, `mixed-sweep.sh`, `quant-quality.sh`, `mtp-depth-sweep.sh`, `mmvq16-test.sh`, `cublas-test.sh`, `faq-test.sh`, `mmvq-profile.sh`, shared setup in `gpu-test-env.sh`).

## 0. Review of the folder before running anything

- `optimize/optimize.py` reproduces `optimize/results.md` byte for byte once `pulp` is present. `pip install pulp` does not work on this Ubuntu 24.04 (PEP 668); `apt install python3-pulp` (2.7.0) does. CLAUDE.md and README now say so.
- Every figure spot-checked in README section 2 traces to a `data/benchmarks.json` cell (ubatch +18%/+4.7%, graphs +7.8%, RCCL 1.31x, topology +3.7%/+1.7%, slot staircase 162/97/153/181, q8_0 KV 32 vs 50). ISA-NOTES line references checked at lines 293 and 7943.
- The client scripts the TODO refers to ("the report-1 client", `mixed-client.py`) live in `/root/rocm-tests/bench/`, not in this folder. `mixed-client.py` takes one URL; the two-server form the TODO's item 3 writes (`--long URL --short URL`) did not exist. Written as `mixed2-client.py`; `dual-server-bench.py` is the two-server form of `server-bench.py`.
- TODO item 1 says `cpupower idle-set -d 2 (disable C6)`. On this host the idle states are 0 POLL, 1 C1, 2 C1E, 3 C6, so `-d 2` disables C1E; C6 is `-d 3`. Both were measured.
- `roc-obj-ls` is not shipped in the TheRock ROCm 7.14 install. `strings /opt/llama.cpp/lib/libggml-hip.so | grep -o 'amdgcn-amd-amdhsa--gfx[0-9a-z-]*' | sort | uniq -c` gives 268 x `gfx906`, no `gfx9-generic`: the build has dp4a.
- `rocm-smi --showrasinfo all` prints empty tables and there is no `ras/features` node under `/sys/class/drm/card*/device/`: RAS is not enabled on these Radeon Pro dies, so there is no ECC cost to measure.
- The folder was zipped on a Mac: `.DS_Store` and a `__MACOSX` sibling came along. Harmless; removed from the folder.

## 1. Small-message RCCL / launch-latency knobs (TODO item 1)

`knobs-sweep.sh`: on top of the guide's baseline environment (`NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16`), each row is `llama-bench` on the four-die tensor split (`-p 2048 -n 256 -d 0 -r 3`) followed by `llama-batched-bench` at batch 8 (2048-token prompts, 128 generated). Baselines were interleaved four times; the three host rows change C-states with `cpupower idle-set` (restored afterwards). RCCL's log confirms each `NCCL_*` variable was taken ("set by environment").

| variant | extra env / host | pp2048 t/s | tg256 t/s | b8 pp t/s | b8 tg t/s |
|---|---|---:|---:|---:|---:|
| base-1 | (baseline) | 846.47 | 47.57 | 847.82 | 159.22 |
| proto-ll | NCCL_PROTO=LL | 811.40 | 47.43 | 812.14 | 164.94 |
| proto-simple | NCCL_PROTO=Simple | 848.50 | 40.46 | 849.05 | 158.88 |
| proto-ll128 | NCCL_PROTO=LL128 | 848.41 | 40.47 | 848.99 | 158.71 |
| algo-ring | NCCL_ALGO=Ring | 848.40 | 47.49 | 849.08 | 158.58 |
| ch16-16 | NCCL_MAX_NCHANNELS=16 NCCL_MIN_NCHANNELS=16 | 848.60 | 47.58 | 849.15 | 152.55 |
| ch32 | NCCL_MIN_NCHANNELS=32 | 846.95 | 47.38 | 847.60 | 155.95 |
| ch8 | NCCL_MIN_NCHANNELS=8 | 849.80 | 47.50 | 850.42 | 157.80 |
| base-2 | (baseline) | 848.75 | 47.60 | 849.15 | 155.10 |
| dev-kernarg | HIP_FORCE_DEV_KERNARG=1 | 848.79 | 47.47 | 849.07 | 158.59 |
| hwq1 | GPU_MAX_HW_QUEUES=1 | 849.00 | 47.45 | 849.13 | 156.29 |
| hwq8 | GPU_MAX_HW_QUEUES=8 | 848.47 | 47.40 | 849.26 | 157.99 |
| no-sdma | HSA_ENABLE_SDMA=0 | 844.25 | 47.53 | 844.74 | 155.74 |
| ll+kernarg | NCCL_PROTO=LL HIP_FORCE_DEV_KERNARG=1 | 812.17 | 47.51 | 812.64 | 160.78 |
| base-3 | (baseline) | 848.44 | 47.56 | 849.25 | 156.22 |
| c6-off | (baseline) host: C6 disabled (cpupower idle-set -d 3) | 847.79 | 47.24 | 847.44 | 155.94 |
| c1e-c6-off | (baseline) host: C1E+C6 disabled (idle-set -d 2 -d 3) | 847.45 | 47.39 | 846.12 | 155.11 |
| ll+kernarg+c6off | NCCL_PROTO=LL HIP_FORCE_DEV_KERNARG=1 host: C1E+C6 disabled | 811.22 | 47.19 | 808.82 | 162.61 |
| base-4 | (baseline) | 849.04 | 47.69 | 849.16 | 157.59 |

Four baselines put single-stream decode at 47.57–47.69 tok/s (0.3% spread) and batch-8 decode at 155.1–159.2 (2.6%). Against that:

- **Nothing moves single-stream decode.** Every environment and host row is within 0.5% of baseline. The 13 ms per token that is not weight streaming is not reachable from the environment; the per-layer synchronisation cost named in README section 5 is code.
- **The default protocol is already the low-latency one.** Forcing `NCCL_PROTO=Simple` costs 15% of single-stream decode; `LL128` gives the same figure because gfx906 has no LL128 path and RCCL falls back to Simple.
- **`NCCL_PROTO=LL` is a batch-8 trade, not a default.** Batch-8 decode is +3.7% (162.8 mean of three LL rows against 157.0 mean of four baselines, about two spreads apart) and prefill is −4.3% in every LL row (811 against 848). It is worth setting only for 8-slot, decode-heavy serving where first-token latency matters less. The serving client re-checks it (section 1a).
- Channel counts (8/16/32, or max=min=16), `NCCL_ALGO=Ring`, `HIP_FORCE_DEV_KERNARG=1`, `GPU_MAX_HW_QUEUES=1/8`, `HSA_ENABLE_SDMA=0`, C6 off, C1E+C6 off: within noise.

`settings/gfx906.env` records this; nothing was added to it.

## 1a. `NCCL_PROTO=LL` in the real server, and `GGML_CUDA_GRAPH_OPT=1` (TODO item 1 follow-up)

`knobs-server-ab.sh`, 10:47–11:03 UTC on the cold-cycled box (clock trace clean, 188 samples). `data/benchmarks.json` `server_knobs_ll_graphopt`.

**(a) LL in `llama-server`** (tp4, `-np 8 -c 262144`, the report-1 client at 1 and 8 clients, base and LL interleaved twice):

| variant | clients | agg gen tok/s | per-request gen tok/s | TTFT s | req/min |
|---|---:|---:|---:|---:|---:|
| base (two runs) | 1 | 33.6 / 33.6 | 46.2 / 46.1 | 1.81 / 1.81 | 7.9 |
| NCCL_PROTO=LL (two runs) | 1 | 33.2 / 33.3 | 46.1 / 46.2 | 1.87 / 1.87 | 7.8 |
| base (two runs) | 8 | 66.1 / 66.1 | 12.8 / 12.8 | 6.39 / 6.38 | 15.5 |
| NCCL_PROTO=LL (two runs) | 8 | 65.9 / 65.8 | 13.0 / 13.0 | 6.59 / 6.59 | 15.4 |

The batched-bench gain (+3.7% at batch 8, §1) does not survive the server: per-request decode is +1.6% (13.0 vs 12.8) but first-token wait is +3% (the −4% prefill), and the aggregate over whole requests is −0.4% at 8 clients and −1% at one. `NCCL_PROTO=LL` stays out of `settings/gfx906.env`; the item-1 verdict ("nothing in the environment moves decode") now holds for the server too.

**(b) `GGML_CUDA_GRAPH_OPT=1`** (rocm0 alone, `llama-bench -p 2048 -n 256 -r 3`, interleaved twice): pp2048 233.0 / 233.1 / 233.0 / 233.2, tg256 19.63 / 19.64 / 19.62 / 19.62. No effect; the pass is a no-op for this graph.

## 2. Two tensor-split pairs (TODO item 2) and the three-die split

`tp2x2-sweep.sh`. The two dies of one MPX module are one XGMI link apart (rocm0+rocm1 and rocm2+rocm3, HIP order), so a pair's allreduce never crosses the bridge.

**A. Batched bench, per pair** (`llama-batched-bench -sm tensor -fa on -b 2048 -ub 2048 -c 71680 -npp 512,2048,8192 -ntg 128 -npl 1,2,4,8`, f16 KV). Pair 0-1 alone, then both pairs at the same time:

| prompt | batch | pair alone: prefill / decode | both: pair01 | both: pair23 | box aggregate (both) | tp4 at the same total streams (report 1 / ladder) |
|---:|---:|---:|---:|---:|---:|---|
| 2048 | 1 | 438 / 31.0 | 437 / 30.9 | 445 / 31.1 | 882 / 61.9 | batch 2: 786 / 70.6 |
| 2048 | 2 | 439 / 54.7 | 438 / 54.4 | 447 / 55.1 | 885 / 109.5 | batch 4: 831 / 121.1 |
| 2048 | 4 | 440 / 80.0 | 439 / 79.7 | 448 / 81.3 | 887 / 161.0 | batch 8: 837 / 159.0 |
| 2048 | 8 | 440 / 93.9 | 439 / 93.7 | 448 / 95.2 | 887 / 188.9 | batch 16: 838 / 150.9 |
| 8192 | 8 | 426 / 89.0 | 426 / 88.9 | 434 / 90.2 | 860 / 179.1 | batch 8 at 8K: 811 / 152.2 |

The pairs do not interfere: pair 0-1 is 93.9 tok/s alone and 93.7 beside the other pair (0.2%), prefill identical. Pair 2-3 is consistently 2% faster than pair 0-1 (the Slot-1 module, the cooler one). At 16 streams the two pairs give 189 tok/s against 151 for tp4 at batch 16 (+25%) and 887 tok/s of prefill against 838; at 8 streams the pairs tie tp4's batch 8 (161 vs 159) with half the per-stream rate of tp4's 8 slots (20.1 vs 19.9 per stream — the same per stream, in fact, since tp4 at batch 8 is also 20 per stream); below 8 streams tp4 wins.

**B. Three dies** (`llama-bench --device rocm0/rocm1/rocm2 -sm tensor -p 2048 -n 256 -d 0,32768 -r 2`, tp4 in the same session): tp3 loads and runs.

| dies | pp2048 | tg256 | pp2048 @ 32K | tg256 @ 32K |
|---:|---:|---:|---:|---:|
| 3 | 596 | 36.4 | 413 | 33.0 |
| 4 | 846 | 47.3 | 636 | 44.2 |

The report's cost model (2.6 ms + 94 µs × 64 layers + 0.493 ms/GiB × 27 GiB × 4/3) gives 26.3 ms, 38 tok/s; measured 27.5 ms.

**C. Servers** (report-1 client: 1300-token prompts, 256 generated, `ignore_eos`; the 2 × tp2 layout is two `llama-server -np 8 -c 262144` processes with clients spread evenly by `dual-server-bench.py`):

| layout | clients | agg gen tok/s | per-request gen tok/s | TTFT s | per-request wall s |
|---|---:|---:|---:|---:|---:|
| 2 × tp2 np8 | 2 | 41.0 | 30.0 | 3.3 | 12.1 |
| 2 × tp2 np8 | 8 | **75.2** | 15.6 | 7.9 | 26.7 |
| 2 × tp2 np8 | 16 | **79.4** | 8.0 | 10.6 | 50.8 |
| tp4 np8 | 1 | 32.7 | 46.3 | 1.8 | 7.8 |
| tp4 np8 | 8 | 66.1 | 12.8 | 6.4 | 30.9 |
| tp4 np8 | 16 | 66.6 | 11.7 | 5.9 | 55.4 |
| tp4 np16 | 8 | 64.8 | 13.3 | 7.8 | 31.4 |
| tp4 np16 | 16 | 63.2 | 5.5 | 7.4 | 64.6 |

In the server the pairs win already at 8 clients (+14%) and by 19–26% at 16, with shorter per-request wall time; the price is 1.5 s more first-token wait at 8 clients because a pair reads a prompt at 440 tok/s. At 2 clients tp4 wins (report 1: 45.6 vs 41.0 here). `tp4 -np 16` is no better than `-np 8` at 16 clients (63 vs 67): the 16-wide MMQ tile costs what the extra slots gain.

**Guide changes.** `optimize.py` gained the `tp2x2` placement (measured cells at 2K and 8K for 1/2/4/8 slots per pair, the tp2 slope beyond); it now picks two pairs for the busy-server profile (179 tok/s at 16 streams, 11.2 per stream, against 153 for tp4 `-np 8`). README sections 2, 3 and 5 updated. The three-die split is recorded in `tp3_llama_bench`. A fix went in alongside: the LP's topology, graphs and draft factors were keyed by slot count only and so credited tp4's +3.7% topology gain to every placement; they are now keyed by placement as well. No winner changed because of that fix.

## 10a. MMVQ instruction counts on gfx906 (TODO item 10, static half)

`mmvq.cu` compiled to gfx906 assembly with the installed build's exact flags (`clang -x hip ... -O3 -mllvm -amdgpu-sched-strategy=max-ilp --offload-arch=gfx906 --cuda-device-only -S`); `isa-count.py` finds each `mul_mat_vec_q<Q8_0, ncols, ...>` kernel, its largest loop (the K loop over 32-weight blocks) and the instruction classes inside it. Each thread handles 8 int8 weights (vdr = 2 dwords) of one block per column per row per iteration; the GCN table gives 2 warps and 1 row per block at one column, 1 warp and 2 rows for 2–8 columns.

| kernel | VGPRs | main loop instructions | VALU | of which v_dot4 | VMEM loads | s_waitcnt | per column × row |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q8_0, 1 column | 19 | 25 | 15 | 2 | 4 | 3 | 25.0 |
| Q8_0, 2 columns (2 rows) | 32 | 55 | 37 | 8 | 8 | 7 | 13.8 |
| Q8_0, 4 columns | 48 | 93 | 66 | 16 | 12 | 11 | 11.6 |
| Q8_0, 8 columns | 64 | 161 | 122 | 32 | 20 | 15 | 10.1 |
| Q8_0, 16 columns (patched build) | 113 | 292 | 234 | 64 | 36 | 18 | 9.1 |

What it says:

- The per-(column × row) cost falls from 25 instructions at one column to 10 at eight, so the kernel does amortise the weight loads; dot4 is 13% of the VALU stream at one column and 26% at eight. The rest is scale conversion and multiply (the f16 `d` of every Q8_0 block and the `ds` of every Q8_1 activation block, converted and multiplied per column), address arithmetic and the loads themselves.
- Instruction issue does not explain the batch-8 step. Per die at batch 8 the loop executes about 2.7 × 10¹¹ lane-instructions per token step (8.5 × 10⁸ blocks × 322), which is 38 ms at the 7.1 T lane-instructions/s VALU peak; the measured single-die step is 155 ms. So VALU is about 25% busy, in line with the ISA-NOTES estimate from the dp4a side.
- What changes between one and eight columns is occupancy and memory-level parallelism: 19 → 64 VGPRs takes a SIMD from 10 resident waves to 4, and each iteration issues 20 loads (12 of them the activation blocks, which are L2-resident but still cost issue slots and L1 bandwidth: 6 KB of activation reads per wave-iteration against 1.3 KB of weights) before a chain of 15 `s_waitcnt`s. The 8-column loop is a latency chain run by 4 waves per SIMD. The rocprof counters (section 10b, when it runs) measure this directly: `SQ_WAIT_INST_ANY`/`SQ_ACTIVE_INST_VALU` against `GRBM_GUI_ACTIVE`, and `SQ_INSTS_VMEM_RD`.
- The 16-column instantiation costs 9.1 instructions per column × row (a further 10% amortisation) but needs 113 VGPRs, which leaves 2 waves per SIMD. Whether that beats the 16-wide MMQ tile is section 9's measurement.

Candidate code changes this points at, none tried yet: fewer VGPRs at 8 columns (1 row per block, or fp16-packed scale math) to get back to 5–6 waves per SIMD; software-pipelining the loads one iteration ahead so the waitcnt chain overlaps the previous iteration's dot4s; converting the per-block scales once per block instead of once per column.

## 10b. MMVQ hardware counters on gfx906 (TODO item 10, GPU half)

`mmvq-profile.sh`, 10:43–10:47 UTC after the item-9 rerun: `rocprofv3 --pmc ... --kernel-trace` around `llama-batched-bench` on one die (`-npp 512 -ntg 32`) at batch 1 and batch 8, stock build. Two of the three counter groups landed; the third (`FETCH_SIZE`, `TCC_HIT/MISS`) produced no pass because rocprofv3 aborts at process exit after the second pass (`retired dangling correlation IDs`, rc=1; the benchmark itself completed both times). Aggregated over every `mul_mat_vec_q` dispatch of each run (13,857 and 15,906: the decode-phase matmuls; prefill goes through `mul_mat_q`). `data/benchmarks.json` `mmvq_rocprof_counters_gfx906`.

| per `mul_mat_vec_q` dispatch, one die | batch 1 | batch 8 | ratio |
|---|---:|---:|---:|
| VGPRs | 20 | 64 | |
| waves | 14,100 | 11,900 | 0.84 |
| VALU instructions per wave | 157 | 637 | 4.1 |
| VMEM read instructions per wave | 29.3 | 96.0 | 3.3 |
| LDS instructions per wave | 5.7 | 34.9 | 6.1 |
| SALU / SMEM instructions per wave | 65 / 9.0 | 97 / 8.3 | 1.5 / 0.9 |
| GPU cycles per dispatch | 184 K | 413 K | 2.25 |
| **VALU busy** (SQ_ACTIVE_INST_VALU × 4 / 256 SIMDs / GRBM_GUI_ACTIVE) | **20%** | **30%** | 1.5 |
| LDS-wait cycles per VALU-active cycle | 0.2% | 0.5% | |
| dispatch duration, mean / median (profiled) | 102 / 53 µs | 239 / 247 µs | 2.3 / 4.6 |

Reading. The counters confirm §10a rather than overturn it: eight columns cost 4.1× the VALU instructions per wave (the static count said 161 vs 25 per iteration, 6.4×; the difference is the fixed per-wave prologue) on 16% fewer waves, and the dispatch takes 2.25× as long, so the SIMDs go from 20% busy to 30%. LDS waits are nowhere (0.5% of VALU-active cycles), which rules out the LDS-crossbar reductions as the stall; what is left in the idle 70% is waiting on the 96 loads per wave (3.3× batch 1's 29) that the 15-deep `s_waitcnt` chain of §10a serialises, at the 4-waves-per-SIMD occupancy that 64 VGPRs allow. The memory-side counters that would have priced the activation re-reads (`FETCH_SIZE`, L2 hit rate) were not obtained; rerunning the third group as its own rocprofv3 invocation would get them.

Consequence for the code change in README §5 item 2: the target is occupancy and load scheduling, not instruction count. Fewer VGPRs at 8 columns (one row per block, or fp16-packed scale math) to get 6–8 waves per SIMD, and loads issued an iteration ahead so the waitcnt chain overlaps the previous iteration's dot4s; the payoff bound is the 70% idle VALU.

The batch-12 kernel trace from the same job explains item 9 mechanically: on the stock build 92% of the `mul_mat` dispatches of a batch-12 single-die run are `mul_mat_q` tiles (9,168 vs 771 `mul_mat_vec_q`), on the patched build 85% are `mul_mat_vec_q` (8,451 vs 1,488).

## 10c. The multi-column MMVQ kernel, iterations 1 and 2 (README §5 item 2)

Started 11:47 after the counters (§10b). The kernel's gfx906 table got two compile-time knobs, rows per thread block for 2–16 columns and warps per block for 5–16 columns, and the Q8_0 path got a fast path that loads each weight block's quants and scale once per row and each activation block's once per column per iteration and applies the two scales as one product per row-column pair. Ten builds (`/opt/llama.cpp-mmvq-*`, names rXwY = rows X / warps Y, f- = fast path; patch `patches/mmvq-gfx906-knobs-and-q8-fastpath-b10288.patch`) measured 13:07–13:53 in one session with stock and the batch-16 build as controls; every patched build passes `test-backend-ops` MUL_MAT 1186/1186; clock trace clean over 553 samples. `data/benchmarks.json` `mmvq_gfx906_variants_iter1`. Aggregate decode tok/s, change against stock:

| build | tp4 b1 | tp4 b2 | tp4 b4 | tp4 b8 | tp4 b12 | tp4 b16 | one die b1 | one die b4 | one die b8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock b10288 | 45.5 | 74.5 | 117.1 | 160.9 | 122.3 | 152.3 | 19.5 | 48.3 | 52.7 |
| mmvq16 (rows 2, warps 1) | 45.8 | 68.7 | 117.4 | 160.6 | 176.6 (+44%) | 171.3 (+12%) | 19.4 | 48.2 | 52.7 |
| rows 1, warps 1 | 45.3 | 58.9 (−21%) | 105.7 (−10%) | 123.9 (−23%) | 119.0 | 113.5 (−25%) | 19.4 | 38.5 (−20%) | 37.2 (−29%) |
| rows 2, warps 2 | 45.3 | 65.9 | 117.6 | 139.0 (−14%) | 161.8 (+32%) | 153.7 | 19.4 | 48.2 | 43.5 (−17%) |
| rows 1, warps 2 / warps 4 | 45.3 | 59–64 | 106 | 111–117 (−29%) | 100 (−18%) | 99–109 (−31%) | 19.4 | 38.5 | 35–36 (−33%) |
| **fast path, rows 2, warps 1** | 45.6 | 72.5 | 118.5 | **172.1 (+7%)** | **196.9 (+61%)** | **207.3 (+36%)** | 19.4 | 49.2 | **59.7 (+13%)** |
| fast path, rows 1, warps 2 | 45.3 | 61.1 | 102.7 | 114.6 (−29%) | 101.1 | 98.1 (−36%) | 19.5 | 37.1 | 33.5 (−36%) |
| rows 4, warps 1 | 45.5 | 68.2 | 115.7 | 170.1 (+6%) | 166.3 (+36%) | 176.1 (+16%) | 19.5 | 51.1 (+6%) | **70.5 (+34%)** |
| fast path, rows 4, warps 1 | 45.5 | 64.5 | 118.5 | 172.6 (+7%) | 170.0 (+39%) | 172.1 (+13%) | 19.4 | 54.3 (+12%) | **71.6 (+36%)** |

What it says. (1) The occupancy hypothesis of §10a/§10b is wrong as a lever: every rows-1 variant, which halves registers and doubles waves per SIMD, loses 20–35% at batch 4 and above on both placements, and more warps per block lose too. Halving the reuse of each loaded activation block costs more than the occupancy buys, so the kernel is bound by the number of load instructions per dot product, not by waves in flight. (2) The fast path on the upstream shape is the tensor-split winner: +7% at batch 8, +61% at 12, +36% at 16 against stock, and +7/+11/+21% against this morning's batch-16 build, from fewer instructions between the loads and the accumulate. (3) Rows 4 is the single-die lever: +34–36% at batch 8 on one die (each activation load now feeds four rows), +6–12% at batch 4, and on the split +6–7% at batch 8 with a smaller gain at 12–16 than the fast path alone. (4) Batch 2 moves by −3% to −13% across builds including the batch-16 build, whose two-column kernel is byte-identical to stock; it is a three-second run and is repeated in the next bench before it is read as a regression.

**Iteration 3** (13:56–14:46, `data/benchmarks.json` `mmvq_gfx906_variants_iter3`, clock trace clean): rows 8, and a whole-block variant of the fast path in which each thread takes a full 32-weight block per iteration (vdr 8: nine aligned dword loads with a funnel shift for the 34-byte weight block, nine dwords for the activation block), a quarter of the load instructions per dot product; the iteration-2 winner and the stock build repeated as controls.

| build | tp4 b1 | tp4 b8 | tp4 b12 | tp4 b16 | one die b1 | one die b4 | one die b8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock (end-of-run repeat) | 45.6 | 161.1 | 122.3 | 153.0 | 19.5 | 48.3 | 52.7 |
| fast path, rows 2 (repeat) | 45.3 | 171.8 (+7%) | 196.6 (+61%) | 206.8 (+35%) | 19.5 | 49.3 | 59.8 (+13%) |
| rows 8 / fast path + rows 8 | 45.5 | 142 (−12%) | 157 (+28%) | 99–141 (−35 / −8%) | 19.4 | 56 (+17%) | 63 (+20%) |
| whole-block loads, rows 2 | **46.5 (+2%)** | 109.5 (−32%) | 110.0 (−10%) | 117.0 (−24%) | **20.4 (+5%)** | 43.0 (−11%) | 37.8 (−28%) |
| whole-block loads, rows 4 | 46.6 (+2%) | 134.6 (−16%) | 145.7 (+19%) | 182.2 (+19%) | 20.4 (+5%) | 52.9 (+9%) | 56.4 (+7%) |

The whole-block hypothesis fails where it matters: a quarter of the load instructions costs a four-times-longer dependent dot-product chain per thread, and at batch 8 that loses 16–32% on the split and 28% on one die. It wins only at batch 1 (+2% split, +5% one die), the single-column case with no reuse to lose. Rows 8 loses to register pressure on the split and gains less than rows 4 on one die. The fast path reproduces to 0.3%. Batch 2 is a noisy three-second cell (the stock control read 74.5 in the first bench and 65 in this one), so no build is judged on it. The model after three iterations: the kernel is bound by activation re-read traffic and the per-thread dependent chain, not by occupancy and not by load count alone; the lever is sharing one loaded activation set across more rows without adding per-thread accumulators.

**Iteration 4** (14:47–15:14, `mmvq_gfx906_variants_iter4`, clock trace clean): the warps of a thread block split the rows instead of K, and each iteration's activation blocks are staged in LDS once per block and read by every warp, so the re-read traffic per row drops by the warp count while each wave keeps the fast path's two-row, 16-accumulator footprint. Four warps × two rows: tp4 batch 8 −16%, batch 16 −35%, one die batch 8 −9%. Two warps × two rows: −32%, −52%, −30%. It shares exactly the activation set that rows 4 in registers shares (+36% on one die), but pays two barriers, a cooperative copy and LDS reads per iteration, and loses; so activation re-read traffic is not the binding resource either. The iteration-2 winners reproduced a third time (172.5 / 197.3 / 207.7 on the split at batches 8 / 12 / 16; 71.7 on one die at batch 8).

The rule after four iterations, from what won and what lost: fewer instructions per dot product wins (the fast path; rows 4, which amortises the per-column loads and conversions over more rows), anything that adds a synchronisation or lengthens the per-thread dependent chain loses (LDS staging, whole-block loads), and register growth beyond about 64 VGPRs loses (rows 8; rows 4 at 12–16 columns). **Iteration 5** (15:46–16:20, `mmvq_gfx906_variants_iter5`): three instruction-count reductions on the fast path, hybrid rows (4 up to 8 columns, 2 for 9–16, the best of both measured shapes), weight quants read as aligned dwords plus a funnel shift instead of half-word pairs, and the two activation dwords a thread needs read as one 8-byte load.

| build | tp4 b1 | tp4 b4 | tp4 b8 | tp4 b12 | tp4 b16 | one die b1 | one die b4 | one die b8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock | 45.6 | 116.8 | 160.9 | 122.2 | 152.3 | 18.9 | 48.4 | 52.8 |
| **fast path, hybrid rows (fh)** | 45.5 | 118.1 | **174.1 (+8%)** | **197.7 (+62%)** | **207.2 (+36%)** | 19.5 | **55.0 (+14%)** | **72.2 (+37%)** |
| fh + aligned weight loads (branch on parity) | 39.1 (−14%) | 106.8 (−9%) | 135.1 (−16%) | 162.2 (+33%) | 167.0 (+10%) | 18.5 | 49.1 | 55.0 |
| fh + aligned weight loads + 8-byte activation loads | 38.7 (−15%) | 105.5 (−10%) | 134.2 (−17%) | 161.8 (+32%) | 166.8 (+9%) | 18.4 | 49.0 | 54.7 |
| fast path, rows 2 (repeat) | 45.6 | 118.9 | 172.4 (+7%) | 197.1 (+61%) | 207.7 (+36%) | 19.4 | 49.4 | 60.0 (+14%) |
| fast path, rows 4 (repeat) | 45.5 | 118.2 | 173.0 (+8%) | 171.0 (+40%) | 172.3 (+13%) | 19.3 | 54.3 (+12%) | 71.6 (+36%) |

The hybrid is the unified build: every cell within 1% of the better of the two earlier winners, on both placements, from one binary. The aligned weight loads as first written lose 9–17% at batches 1–8: the branch on the block's parity is divergent within a wave (the parity alternates across lanes), so both paths execute; the 8-byte activation load adds nothing because the compiler already merges the two dword loads. A branchless version with a dynamic funnel-shift amount (iteration 5b, 17:47–18:05, `mmvq_gfx906_variants_iter5b`) does not help either: −2% at batch 8, −10% at 12 and −16% at 16 on the split against the hybrid build, because the three live dwords per row cost registers exactly where 12–16 columns are register-critical; +3% / −3% on one die. The aligned-load knob is dropped. The fast path reproduced a fifth time (174.1 / 198.2 / 207.0 on the split at 8 / 12 / 16; 54.4 / 71.7 on one die at 4 / 8); batch 2 is noise (the stock control read 58, 60 and 67 in the last three runs after 74, 65, 66 and 65).

**Where this leaves the kernel.** Against the stock build, the `fh` binary gives the four-die split +8% at 8 slots, +62% at 12 and +36% at 16, and a single die +14% at 4 slots and +37% at 8, with correctness on 1186 of 1186 matrix-multiply tests in every build. That is the guide's new build for Qwen3.8-27B Q8_0 serving: `/opt/llama.cpp-mmvq-fh`, patch `patches/mmvq-gfx906-knobs-and-q8-fastpath-b10288.patch` with `GGML_MMVQ_GCN_ROWS 4`, `GGML_MMVQ_GCN_ROWS_HI 2`, `GGML_MMVQ_GCN_Q8_FASTPATH 1`. ISA-NOTES s.3's target of "an 8-column step at ~1.3× the single-column cost" (about 270 tok/s at 8 slots) is not reached; at 8 slots the step is 2.1× (174 tok/s), and the measurements say the remaining distance is not occupancy, not load count and not activation traffic but the per-wave instruction stream itself, which points at the Q8_1 activation quantisation format (a half2 scale pair per 32 values that has to be converted per column per block) and at packed-fp16 scale math as the next things to try, with a rocprof re-count of the `fh` kernel to see what the 30% VALU-busy figure became.

## 6. Model-file quality: Q4_0, Q4_K_M, Q6_K against Q8_0 (TODO item 6)

`quant-quality.sh`: `llama-perplexity` on wikitext-2 test, `-c 16384 --chunks 6` (as report 2 section 4), four-die tensor split. The Q8_0 run saved its logits (`--kl-divergence-base`, 23 GB); each other file was scored against them (`--kl-divergence`). The Q8_0 perplexity came out 5.597 ± 0.062 against 5.617 in report 2, within its error.

| file | GiB | PPL | PPL / Q8_0 | mean KLD (nats) | median KLD | top token unchanged | speed on four dies (report 2) |
|---|---:|---:|---:|---:|---:|---:|---|
| Q8_0 | 27.05 | 5.597 | 1 | 0 | 0 | 100% | decode 45.7, prefill 834 |
| Q4_0 | 14.95 | 6.167 | **1.105** | **0.091** | 0.013 | **91.6%** | +11% decode, +33% prefill |
| UD-Q4_K_M | 15.33 | 5.501 | 0.985 | 0.036 | 0.004 | 95.3% | +4% decode, −12% prefill |
| UD-Q6_K | 20.47 | 5.626 | 1.008 | 0.019 | 0.001 | 97.6% | 0% decode, −22% prefill |

- **Q4_0 is a real quality loss**: +10.5% perplexity and a different top token on 8.4% of positions. The +11% decode / +33% prefill in README section 2 stays a quality trade; the `--allow-quant` runs of `optimize.py` should be read with this table beside them.
- **Q4_K_M is the 4-bit file if 4-bit is wanted.** Its perplexity is below Q8_0 on this text (an imatrix effect; the "UD" files were calibrated on similar prose), which is why KL divergence rather than perplexity is the judge: 0.036 nats, 2.5× less than Q4_0. On Vega 20 it costs 12% of prefill for 4% of decode.
- **Q6_K is a capacity trade, not a speed one**: 0.019 nats from Q8_0 for equal decode and 22% less prefill. On the four-way split it frees only 1.6 GiB per die (7% more context per slot); on single-die instances it frees 6.6 GiB, which turns 4K per slot at eight slots into 17K (table below).
The capacity side of a smaller file (memory model of `optimize.py`, 31 GiB per die, f16 cache, `-ub 2048`): largest context per slot that fits.

| layout | slots per instance | Q8_0 | Q6_K | Q4_K_M | weights freed per die by Q6_K |
|---|---:|---:|---:|---:|---:|
| tp4 | 8 | 175K | 188K | 199K | 1.6 GiB |
| 2 × tp2 | 8 | 57K | 70K | 80K | 3.3 GiB |
| 2 × tp2 | 4 | 116K | 143K | 163K | 3.3 GiB |
| four single dies | 8 | 4K | 17K | 27K | 6.6 GiB |
| four single dies | 4 | 10K | 36K | 57K | 6.6 GiB |
| four single dies | 1 | 49K | 154K | 237K | 6.6 GiB |

On the four-way split the weights are already quartered, so Q6_K buys 7% more context for 22% less prefill; on the pairs it buys a quarter more; on single-die instances, where 25.8 GiB of Q8_0 leaves 4K per slot at eight slots, Q6_K quadruples the context per slot at 0.019 nats. `optimize.py --allow-quant` sees this through `memory_gib` and will pick Q6_K when a single-die or pair workload is capacity-bound.

- **Q4_1 (same repo, 16.33 GiB), measured 11:03–11:10** (`q41-quality.sh`, clock trace clean; `data/benchmarks.json` `quant_quality_q4_1`): PPL 5.463 against 5.583 for Q8_0 on the same chunks (0.979), mean KLD **0.056**, median 0.007, top token unchanged **93.7%**. Speed, same session (`llama-bench -p 2048 -n 256 -r 2`):

| file | rocm0 pp2048 | rocm0 tg256 | tp4 pp2048 | tp4 tg256 |
|---|---:|---:|---:|---:|
| Q4_1 | 322 | 27.5 | 1129 | 53.5 |
| Q4_0 | 332 | 27.1 | 1137 | 52.9 |
| Q8_0 | 234 | 20.2 | 849 | 47.3 |

  So Q4_1 is Q4_0's speed (within 3%) with 40% less divergence (0.056 vs 0.091), but still 1.6× Q4_K_M's 0.036; Q4_K_M keeps the "4-bit file if 4-bit is wanted" line, and Q4_1 is the choice only where prefill matters more than 0.02 nats (Q4_K_M is −12% prefill). `optimize.py --allow-quant` knows Q4_1 (factors from these rows).

## 3. Isolating short traffic from a long prefill (TODO item 3)

`mixed-sweep.sh` with `mixed2-client.py`: one 130,816-token prompt and seven 4,096-token prompts sent at the same instant, 256 tokens generated each, `--cache-ram 0`, the long request routed to one server and the short ones to another. The control is report 2's setup re-measured in the same session.

| layout | long: first token s | long: gen tok/s | long: wall s | short: first token mean / max s | short: gen tok/s | short: wall mean / max s |
|---|---:|---:|---:|---:|---:|---:|
| control: one tp4 `-np 8` (8 × 128K) takes all eight | 262 | 13.0 | 320 | 17.7 / 44.0 | **0.9** | 318 / 319 |
| A: long → tp2 pair 2-3 (`-np 2`, 2 × 256K); short → tp2 pair 0-1 (`-np 8`, 8 × 32K) | **476** | 23.1 | 487 | 29.2 / 70.0 | **8.1** | **90 / 90** |
| B: long → tp3 (`-np 4`, 4 × 128K); short → one die (`-np 6`, 6 × 4608) — rerun 11:10–11:17, clock trace clean | **401** | 25.5 | 412 | 42.5 / 99.0 | **5.9** | **155 / 170** |

The control reproduces report 2 section 5 to the second (264 / 0.9 / 320 there). Giving the short traffic its own pair does what the report predicted: the seven short requests finish in 90 s at 8.1 tok/s each instead of 318 s at 0.9. The long request pays: a pair reads the prompt at 287 tok/s, so its first token comes at 476 s instead of 262, and tp4 reads it at 499 tok/s even while interleaving seven short prompts. The TODO's "short ≥ 10 tok/s and long first token under 400 s" is not reachable on four dies at once. Layout B, measured at 11:17 (`data/benchmarks.json` `server_mixed_wave_isolated.layout_B_rerun`), gets the long prompt's first token to 401 s (three dies read it at ~326 tok/s) but holds the seven short requests to 5.9 tok/s and 155 s of wall time on their single die, against 8.1 tok/s and 90 s on a pair in layout A. So: A when short traffic dominates, B when the long prompt's first token matters more; neither reaches both targets.

The operational rule this gives: route by prompt length at the proxy, keep the long-prompt instance the bigger split, and accept that a 128K prompt costs 4–8 minutes of first-token wait whatever the layout. If the box mostly serves short traffic with rare long prompts, layout A (two pairs) is also the busy-server winner of section 2, so it costs nothing to run permanently.

## 4. `--kv-unified` on a newer llama.cpp (TODO item 4)

Upstream tag b10837 (commit 5202104, 2026-09-07; 549 tags after our b10288) built 16:33 with the SETUP.md recipe into `/opt/llama.cpp-b10837` (its `--version` string wrongly reports the b10288 commit; the ggml library is 0.23.0 against 0.18.1, which is the proof of the source). `kvu-rerun-b10837.sh`, 16:35–17:35, cold-cycled box, clock trace clean. `data/benchmarks.json` `kv_unified_b10837`. One 261,888-token prompt alone, 256 generated:

| server / bench | prefill tok/s | first token s | decode tok/s |
|---|---:|---:|---:|
| per-slot server, 4 × 256K (b10288: 366 / 716) | **395** | 663 | 30.8 |
| `--kv-unified` pool, 8 × 256K (b10288: 188 / 1393) | **394** | 664 | 30.8 |
| batched bench, private, 1 sequence | 394.9 | 663 | 30.9 |

The pooled server now reads the prompt at the bench's rate: the prompt-path penalty of report 2 s.5 (half speed through the pool) is gone upstream, and the private prompt path is 8% faster than on b10288 as well (395 vs 366 tok/s, 53 s less to the first token at 256K). What has not been re-measured is the pool's decode cost at 8 × 32K, which lost a third on b10288 in the batched bench; until it is, `--kv-unified` is a capacity mode for rare long requests rather than a prompt-path loss. The guide's numbers stay on b10288; this is the one table measured on the newer build, and the 8% prompt-path gain is the first reason to consider moving the baseline.

## 5. MTP draft length at 4 and 2 slots with depth (TODO item 5)

`mtp-depth-sweep.sh`: `llama-server` on the four-die split, `-np N -c N × (depth + 2048)`, greedy, 300 tokens generated; `mtp-depth-client.py` sends N distinct wikitext slices of `depth` tokens with a summary instruction at the same instant. "per-request gen" is the server's own `predicted_n / predicted_ms`.

| variant | depth | slots | first token mean / max s | per-request gen tok/s | sum of per-request | acceptance |
|---|---:|---:|---:|---:|---:|---:|
| none | 32K | 4 | 76 / 144 | 8.9 | 35.7 | |
| draft 1 | 32K | 4 | 80 / 153 | 9.7 | 38.8 | 0.88 |
| draft 2 | 32K | 4 | 82 / 157 | 8.6 | 34.2 | 0.79 |
| none | 32K | 2 | 52 / 53 | 17.2 | 34.4 | |
| draft 2 | 32K | 2 | 54 / 54 | 25.1 | 50.2 | 0.79 |
| draft 3 | 32K | 2 | 54 / 55 | 25.0 | 49.9 | 0.69 |
| none | 128K | 4 | 397 / 782 | 5.9 | 23.4 | |
| draft 1 | 128K | 4 | 426 / 841 | 9.3 | 37.2 | 0.97 |
| draft 2 | 128K | 4 | 429 / 845 | 11.3 | 45.4 | 0.90 |
| none | 128K | 2 | 265 / 271 | 12.7 | 25.3 | |
| draft 2 | 128K | 2 | 283 / 291 | 22.4 | 44.9 | 0.93 |
| draft 3 | 128K | 2 | 284 / 292 | 26.0 | 51.9 | 0.88 |

Two things are solid. Acceptance rises with depth on this kind of prompt (a summary of 128K of prose is predictable): 0.88 → 0.97 for draft 1, 0.79 → 0.90 for draft 2. And at 32K the picture is the one the guide already has: at 4 slots draft 1 is +9% and draft 2 is −4%; at 2 slots draft 2 and 3 are +45%. Drafting adds 4–8% to the wave's first-token time.

**Decode-only rerun, 11:18–13:06** (`mtp-depth-sweep2.sh`, two waves: the second re-sends the prompts with `cache_prompt` on so all slots decode together; clock trace clean over 1292 samples; `data/benchmarks.json` `speculative_mtp_depth_np_two_wave`). Per-request generation from the server's own timings, wave 2 (aggregate = slots × per request):

| depth | slots | none | draft 1 | draft 2 | draft 3 |
|---:|---:|---:|---:|---:|---:|
| 32K | 4 | 24.2 (96.9) | 26.9 (107.6, **+11%**) | 21.2 (84.8, −12%) | |
| 32K | 2 | 32.5 (65.1) | | 44.6 (89.3, +37%) | 45.8 (91.7, **+41%**) |
| 128K | 4 | 16.7 (66.8) | 20.1 (80.4, **+20%**) | 17.7 (70.8, +6%) | |
| 128K | 2 | 25.7 (51.4) | | 38.8 (77.7, +51%) | 41.5 (82.9, **+61%**) |

The burst rows above overstated the 4-slot gains at 128K (+61% / +106% for drafts 1 / 2 become +20% / +6% decode-only) for the reason given; at 32K the two protocols agree. Acceptance rises with depth (0.88 → 0.98 for draft 1 at 4 slots). The wave-2 wall clocks are not used: the server queued some wave-2 requests behind others (a 25 s wave for 12.6 s requests), which the per-request timings are immune to. Guide: the `long` profile keeps draft 1 at 4 slots (+20% at 128K); two slots with draft 3 is the fastest per-stream long-context setting. `optimize.py` now carries depth-dependent draft factors (32K and 128K points, interpolated, held beyond).

The original paragraph stands as the reason for the rerun: the 128K burst rows overstate the steady-state gain. With four 128K prompts arriving together the first request generates its 300 tokens while the other three are still being read (first token 397 s mean, 782 s max), so every decode step of the no-draft baseline is interleaved with prefill micro-batches; a drafted run needs fewer steps and so suffers fewer of them. That is a real effect for burst arrivals, but the LP's decode cells are all-slots-decoding figures. A second pass with a two-wave protocol (send the prompts once to fill the slot caches, then again with `cache_prompt` on so all slots decode together) is queued as the last job; its wave-2 ratios will replace the short-depth factors for 2 and 4 slots at depth. Until then the `long` profile keeps draft 1 at 4 slots.

## 8. `GGML_CUDA_FA_ALL_QUANTS=ON` build (TODO item 8)

Run 09:58-10:26 UTC on the cold-cycled box (queue-after-reboot.sh job 1; clock trace clean over all 329 samples, longest 1000 MHz-under-load run one sample; CPU RAPL 150 W, four dies at the 200 W cap, 1044 W DC / 1346 W AC). Build `/opt/llama.cpp-faq` (b10288 + the flag; 49 attention-vector instances). `data/benchmarks.json` `fa_all_quants_kv_tp4`.

| 8 sequences, tp4, llama-batched-bench | 2K prefill t/s | 2K decode t/s | 32K prefill t/s | 32K decode t/s |
|---|---:|---:|---:|---:|
| f16 KV | 847.2 | 159.7 | 737.1 | 131.2 |
| q8_0 K / q8_0 V | 845.5 | 153.2 | 733.2 | 111.2 |
| q8_0 K / q4_0 V | 846.1 | 152.8 | 733.4 | 107.4 |

What it settles. (1) The q4_0 value cache runs on the tensor split with this build (the stock build aborts); it costs 3% against q8_0/q8_0 at 32K depth and nothing at 2K, so the "4-bit V cache has no GPU kernel here" line in README s.2 is now build-dependent. (2) q8_0 decode at depth does not close on f16: 111 vs 131 (-15%), the same gap the stock build showed (110.8 in report 2), so the quantised-KV kernels of this build are not faster than the fallback path the stock build used for q8_0. (3) The memory ceiling moves: 8 x 256K fits with q8_0 K / q4_0 V (pp512 848.6, tg32 142.8 t/s at -c 2097152) and does not fit with q8_0/q8_0, which dies allocating the 9536 MiB compute buffer on device 0; the q8_0 cache's compute buffer does not shrink with this build. Guide change: "8 x 256K does not fit at any KV type" (README s.3, launch `long`) becomes "fits with q8_0-K/q4_0-V on the FA_ALL_QUANTS build", as a capacity mode at -18% decode.

Perplexity (rerun 17:26–17:31 as `faq-ppl-rerun.sh` after the job's own rows were lost to a capture bug: the script grepped `^Final estimate` and this build prefixes every log line with a timestamp): **q8_0 K / q4_0 V 5.619 ± 0.062, q8_0 / q8_0 5.611 ± 0.062**, against 5.615 ± 0.062 for f16 in report 2 s.4. All three are within 0.15% of each other, far inside the error, so neither quantised cache, the 4-bit value cache included, costs measurable quality at 16K context. That makes the 8 × 256K capacity mode a free one on quality: −18% decode at depth is its whole price. (The build does not print the context buffer sizes, so the "compute buffer / KV" fields of the job's output stay empty; the 9.5 GiB compute buffer is known from the q8_0/q8_0 out-of-memory line.)

## 9. `MMVQ_MAX_BATCH_SIZE` 8 → 16 (TODO item 9)

Rerun 10:26–10:43 UTC with the library path fixed (`ldd` on the patched binary logged; the 06:33 run had loaded the stock library). Clock trace clean. `data/benchmarks.json` `mmvq16_batched_bench_tp4_2k`; patch saved as `patches/mmvq-max-batch-16-b10288.patch` (mmvq.cu, mmvq.cuh, ggml-cuda.cu; the mmid path keeps 8). Correctness: `test-backend-ops` MUL_MAT 1186/1186 on ROCm0.

**Four dies, tensor split** (`llama-batched-bench -npp 2048 -ntg 128`, patched vs stock in the same session):

| batch | patched decode t/s | stock decode t/s | change | prefill (both) |
|---:|---:|---:|---:|---:|
| 1 | 44.5 | 44.4 | 0% | 835 |
| 8 | 158.2 | 159.5 | −1% | 846 |
| 9 | **162.6** | 96.5 | **+69%** | 846 |
| 10 | 166.7 | — | | 846 |
| 12 | **174.6** | 122.5 | **+43%** | 846 |
| 14 | 171.3 | — | | 846 |
| 16 | **170.7** | 152.5 | **+12%** | 846 |

**One die** (`--device rocm0 -npp 512`):

| batch | patched decode t/s | stock decode t/s | change |
|---:|---:|---:|---:|
| 1 | 19.45 | 19.44 | 0% |
| 8 | 52.7 | 52.8 | 0% |
| 12 | 51.7 | 54.7 | **−6%** |
| 16 | 46.6 | 63.0 | **−26%** |

Sanity on the patched build, tp4 `llama-bench`: pp2048 848.4, tg256 47.2 (stock 846–849 / 47.6).

What it settles. On the tensor split the 9–16 stair is gone: batch 9 no longer falls to the 16-wide MMQ tile (96 → 163 tok/s), 12 slots become the new 2K peak at 175, and 16 slots gain 12%. Batches 1–8 and prefill are untouched. On a single die the opposite happens: the 16-column MMVQ (113 VGPRs, 2 waves per SIMD, §10a) loses to the MMQ tile by 6% at 12 and 26% at 16. The explanation is width: on tp4 each die multiplies a quarter-width matrix, where the latency-bound MMVQ still beats a tile's fixed cost; on a full-width single-die matrix the tile's reuse wins. So the guide change is scoped: **the patched build is the build for tensor-split servers that want 9–16 slots** (`-np 12` at 175 tok/s replaces the "sit at 8 or 16" rule there), and single-die instances stay on the stock build at `-np 8`. `optimize.py --build mmvq16` adds slots 9–15 and re-prices 16 for tp4 (and only tp4: the other placements have no patched cells and keep their stock values).

The counter profile for item 10 (§10b) ran after this in the same job.

## Power-cap sweep: performance per watt (the TODO's future item)

`powercap-sweep.sh`, 15:15–15:45 UTC on the cold-cycled box, CPU RAPL 150 W, clock trace clean; `data/benchmarks.json` `power_cap_sweep`. Per-die cap 200 → 185 → 175 → 160 → 150 W, then 200 again as the drift control. Per point: `llama-bench -p 2048 -n 256 -r 2` on one die and on the four-die split, then the 8-slot server under the report-1 client at 1 and 8 clients. Power is the four dies' `power1_input` averaged over the server phase (the clean four-die decode window) and the SMC's two MPX-bay readings over the same window.

| cap W | one die pp2048 | one die tg256 | tp4 pp2048 | tp4 tg256 | server 8 clients agg tok/s | TTFT s | four dies W (server) | bays W (SMC) | server tok/s per 100 W of die power |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 232.9 | 20.32 | 847 | 47.4 | 65.7 | 6.41 | 614 | 761 | 10.7 |
| 185 | 225.9 | 19.95 | 831 | 47.1 | 65.5 | 6.47 | 655 | 778 | 10.0 |
| 175 | 222.3 | 19.65 | 818 | 47.2 | 64.5 | 6.55 | 551 | 736 | 11.7 |
| 160 | 215.1 | 19.01 | 793 | 46.5 | 63.0 | 6.69 | 532 | 704 | 11.8 |
| 150 | 209.6 | 18.47 | 776 | 45.8 | 61.6 | 6.84 | 531 | 687 | 11.6 |
| 200 (repeat) | 226.7 | 20.28 | 848 | 47.1 | 65.7 | 6.36 | 684 | 747 | 9.6 |

Reading. Decode does not reach the cap: four dies decoding for eight clients draw 530–680 W in total, 130–170 W each, so the cap only binds in prefill. Lowering it to 175 W costs 0.4% of four-die decode, 2% of server throughput at eight clients and 3% of single-die prefill; 150 W costs 3%, 6% and 10%. Bay power at the server point falls from 761 W to 736 (175) and 687 (150); the sampler-window means are ±5–10% noisy (the 200 W repeat read 684 W of die power against 614 the first time at the same throughput), so the per-watt column is indicative: about +8–10% at 160–175 W. The operational point is the envelope, not the watt-hours: **at 175 W per die the four dies at the cap plus a fully loaded host come to about 1164 W DC, inside the 1228 W envelope**, so the clamp risk from host work beside the GPUs goes away for a 2% serving cost. Recommendation: cap the dies at 175 W for serving (`rocm-smi --setpoweroverdrive 175` or `power1_cap` = 175000000 in sysfs on each die) and keep 200 W only for prefill-heavy ingest work. `gpu-test-env.sh` keeps 200 W for benchmarks so that today's numbers stay comparable.

### Sweep 2: 180 to 155 W in 5 W steps on the production build (queue-8, 21:38–22:19)

Requested after the production build landed; `powercap-sweep2.sh`, `power_cap_sweep2`. Same per-point protocol on `/opt/llama.cpp-mxxm-fh`, the server at `-np 16` under 8 and 16 clients, 200 W at both ends. Throughput only; the 200 W repeat reproduced the opening reference within 0.3% on every bench cell and within 2% on the server, so the ladder is clean.

| cap W | one die pp2048 | one die tg256 | tp4 pp2048 | tp4 tg256 | server 8 clients | server 16 clients | TTFT s (16) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 320.7 | 20.29 | 1127 | 48.9 | 77.4 | 81.5 | 5.77 |
| 180 | 310.0 (−3%) | 19.80 (−2%) | 1096 (−3%) | 48.9 | 76.3 (−1%) | 79.5 (−2%) | 5.91 |
| 175 | 306.6 (−4%) | 19.69 (−3%) | 1085 (−4%) | 48.6 (−1%) | 75.3 (−3%) | 78.8 (−3%) | 5.85 |
| 170 | 303.6 (−5%) | 19.52 (−4%) | 1073 (−5%) | 48.6 (−1%) | 75.1 (−3%) | 78.5 (−4%) | 5.87 |
| 165 | 300.0 (−6%) | 19.36 (−5%) | 1060 (−6%) | 48.2 (−1%) | 74.5 (−4%) | 76.8 (−6%) | 6.07 |
| 160 | 295.4 (−8%) | 19.10 (−6%) | 1046 (−7%) | 47.9 (−2%) | 73.3 (−5%) | 77.0 (−6%) | 5.99 |
| 155 | 291.1 (−9%) | 18.85 (−7%) | 1034 (−8%) | 47.4 (−3%) | 72.9 (−6%) | 76.2 (−7%) | 6.05 |
| 200 (repeat) | 321.5 | 20.76 | 1129 | 48.9 | 76.1 | 81.3 | 5.78 |

The production build's prefill leans harder on the cap than stock's did (stock lost 3% at 185 and 10% at 150); four-die decode is flat to 170 W and loses 3% at 155. **The power side of both sweeps is not fit for a per-watt decision**: the 200 W repeat drew 715 W of sampled die power against 603 at the opening point at the same throughput (+18%), while the SMC bay reading moved 2% (753 vs 766 W). Sampled power over a window with prefill bursts and decode plateaus depends on where the 5 s samples land and on die temperature (leakage rises over a 40-minute run). The per-watt question is therefore answered by the adaptive study below, which integrates energy over a fixed workload at 1 s, visits every cap in down-then-up passes, and replicates.

### Adaptive search for the best serving performance per watt (queue-9, from 22:19)

`powercap-adaptive.sh` / `powercap-adaptive.py`, `power_cap_adaptive`. Running; the section is written when it finishes.

## 11. The ML-gfx906 fork's MMQ tile table (the "mxxm" preset)

The ML-gfx906 project ships an alternate llama.cpp preset: `mxxm-t/mx-llama.cpp` at b10254 (2026-08-04) with a 14-line patch to its gfx906 MMQ tile configuration for Q4_K/Q5_K/Q6_K (`build-context/patch/mxxm-gfx906-kcase.patch`, "measured on MI50, +15–35% Q5_K/Q6_K"). The patch targets a file that exists only in that fork, so the fork was built (`build-mxxm.sh`, 17:31, SETUP.md recipe, `/opt/llama.cpp-mxxm`; its `--version` string wrongly reports the b10288 commit) and measured against stock b10288 in one session (`mxxm-bench.sh`, 17:34–17:46, library paths verified, clock trace clean). `data/benchmarks.json` `mxxm_fork_llama_bench`.

| file | one die pp2048 | one die tg256 | tp4 pp2048 | tp4 tg256 |
|---|---:|---:|---:|---:|
| Q8_0, stock b10288 | 233.6 | 20.59 | 848 | 47.3 |
| Q8_0, mxxm fork | **315.0 (+35%)** | 20.78 | **1130 (+33%)** | 49.1 (+4%) |
| Q6_K, stock | 175.0 | 20.99 | 658 | 47.5 |
| Q6_K, mxxm fork | **228.7 (+31%)** | 21.01 | **827 (+26%)** | 49.4 |
| Q4_K_M, stock | 201.3 | 24.86 | 746 | 49.7 |
| Q4_K_M, mxxm fork | 216.7 (+8%) | 24.99 | 796 (+7%) | 51.8 (+4%) |

This is the largest prefill result of the day, and it is not the kcase patch: that touches Q4_K/Q5_K/Q6_K, while the Q8_0 gain comes from the fork's own gfx906 tile table (`mmq-config-gfx906.cuh`, a Q8_0 case with 8 warps and its own I/J tile), which upstream b10288 does not have (its MMQ uses a generic table on gfx906). A third more prompt throughput on the serving model moves every first-token figure in this guide by the same fraction, the 128K prompt of §3 from 262 s toward 200 s, and the ingest profile from 750 toward 1000 tok/s at 32K; decode moves 1–4%. Two things stood between it and the guide: correctness (the fork build has no `test-backend-ops`), and the combination with the MMVQ fast path of §10c. Both are done (queues 5 and 6, 18:06–18:30, `mxxm_fh_batched_bench_tp4_2k`).

**Correctness.** Q8_0 perplexity on the split at -c 16384 (six chunks of wikitext-2): the fork reads 5.5969 ± 0.062, and the fork with the §10c kernel also 5.5969, the stock reference to four digits (5.597). The tile table and the fast path are exact.

**The combined build.** The knob patch applies to the fork's tree once the header half of the 16-column change goes in first (`patches/mmvq-max-batch-16-b10288.patch`, the `ggml-cuda.cu` and `mmvq.cuh` hunks; the knob patch carries the `mmvq.cu` half, which is what made queue-5's build fail on an undeclared constant). `build-mxxm-fh.sh` installs the result as `/opt/llama.cpp-mxxm-fh` (fork + kcase patch + both patches, knobs ROWS 4 / ROWS_HI 2 / FASTPATH 1) and puts the fork's tree back to its own state. Against stock b10288 in the same session (`mxxm-fh-bench.sh`):

| Q8_0 | one die pp2048 | one die tg256 | tp4 pp2048 | tp4 tg256 | tp4 decode at 4 / 8 / 12 / 16 | one die decode at 4 / 8 |
|---|---:|---:|---:|---:|---:|---:|
| stock b10288 | 229.8 | 20.26 | 848 | 47.3 | 107.2 / 159.8 / 122.5 / 152.7 | 48.3 / 52.7 |
| mxxm-fh | **313.6 (+36%)** | 20.52 | **1130 (+33%)** | 48.8 (+3%) | 123.6 (+15%) / 173.6 (+9%) / **198.6 (+62%)** / **204.1 (+34%)** | 53.3 (+10%) / **69.1 (+31%)** |

The decode cells match the fh build on b10288 within 3% (173.6 / 198.6 / 204.1 against 174.1 / 198.2 / 207.0 on the split; 69.1 against 71.7 on one die at batch 8), so the fork's tree costs the kernel nothing, and the fork's prefill carries over unchanged (1130 both ways; 1140 in batched-bench). Request-level throughput on the split (2048 in, 128 out) is +27% at 8 slots, +42% at 12 and +35% at 16. One artefact to know when reading batched-bench: its first batch pays a one-time cost that any build can show (the combined build read 580 tok/s on its first 2048-token prompt and 1114 on repeats; stock read 627 then 839 in the same check, `mxxm-fh-warmup-check.sh`), which llama-bench hides with its warm-up run; a server pays it once, on its first request.

`/opt/llama.cpp-mxxm-fh` is the production build: `optimize.py --build mxxm+fh` re-prices the workloads (results.md; every prefill figure by the fork's ratio, decode as `fh`), README §2/§3 and `settings/launch.sh` point at it, and stock b10288 stays the reference every other number in this report is measured on.

**Server level (queue-7, 18:41–18:52, `mxxm_fh_server_np16`).** The optimiser's 16-slot rows rest on batched-bench cells, so the production build went through `llama-server` (tp4, `-np 16`, 16 × 32K) under the report-1 client (1300-token prompts, 256 generated; aggregate = generated tokens over the wall clock of the whole wave, prompt time included), stock in the same session:

| clients | production build: agg gen / per request / first token | stock b10288: agg gen / per request / first token | two tp2 pairs (§2, same client) |
|---:|---|---|---|
| 4 | 65.1 tok/s / 28.2 / 4.96 s | 57.9 / 27.0 / 6.20 s | |
| 8 | **75.5** / 14.5 / 5.25 s | 65.1 / 13.0 / 6.66 s | 75.2 / 15.6 / 7.91 s |
| 12 | **83.1** / 10.1 / 5.65 s | 58.8 / 6.7 / 7.29 s | |
| 16 | **82.1** / 7.3 / 5.63 s | 63.6 / 5.5 / 7.38 s | 79.4 / 8.0 / 10.56 s |

+16% at 8 clients, +41% at 12 and +29% at 16 over stock on the same server, with first tokens 1.2–1.7 s sooner from the prefill gain. One four-die server at 16 slots now matches the two pairs at 8 clients and beats them at 16 with half their first-token latency, so the busy-server recommendation moves from the pairs to tp4 `-np 16` on the production build (`launch.sh team16`); the pairs remain the answer to head-of-line blocking (§3). Twelve clients is the peak of the curve; sixteen costs nothing in aggregate and trades per-request decode (10.1 to 7.3 tok/s).

## Interruption: the 1000 MHz clamp, 06:47 UTC

After 3.7 hours of continuous load (03:06–06:47, perf level high, fans at full speed, governor performance, jobs toggling the perf level between them) all four dies dropped to 1000 MHz within 40 s of each other during the FORCE_CUBLAS test, while die 0b was at the 200 W cap in a single-die prefill. Under load they then draw ~85 W at 1000 MHz; `pp_dpm_sclk` still lists 1730 MHz but neither `--setperflevel auto/high`, `--resetclocks`, nor `manual` with state 8 pinned moves them (prefill on one die 145 tok/s against 235). The sampler signature is ≥ 6 consecutive 5 s samples at 1000 MHz with more than 60 W; by that signature every run before 06:47 in this report is clean (isolated 1000 MHz samples during model loads are normal), and the FORCE_CUBLAS and FA_ALL_QUANTS runs are not. As on 2026-09-04, a reboot is the only known cure.

Preliminary from the FORCE_CUBLAS run's first two rows, taken before the clamp hit: single-die `pp2048` 241 ± 6 tok/s and `tg256` 20.3 against 235 / 20.3 for the stock build in report 1, which is the "expected to lose or tie" outcome of ISA-NOTES section 3 (rocBLAS fp16 GEMM moves twice the bytes of MMQ's dp4a tiles and lands within noise of it). Confirmed by the second run (§7 below); the four-die rows were never obtained at full clock.

## 7. `GGML_CUDA_FORCE_CUBLAS=ON` build (TODO item 7)

`cublas-test.sh`: `llama-bench -p 2048 -n 256 -d 0 -r 2` on one die and on the four-die split, the `/opt/llama.cpp-cublas` build (rocBLAS fp16 GEMM for every matrix multiply instead of the MMQ dp4a tiles) against the stock build in the same session. Run twice (06:4x and 08:12); both times the clock clamp hit right after the single-die rows.

| build | run | rocm0 pp2048 | rocm0 tg256 | tp4 pp2048 | tp4 tg256 |
|---|---|---:|---:|---:|---:|
| FORCE_CUBLAS | 08:12, before the clamp | 245.2 ± 0.1 | 20.15 | 771 (clamped) | 34.0 (clamped) |
| FORCE_CUBLAS | 06:4x, before the clamp | 241 ± 6 | 20.3 | — | — |
| stock | report 1 | 235 | 20.25 | 834 | 45.7 |
| stock | 08:14, clamped control | 145.0 | 15.6 | 538 | 33.9 |

On one die at full clock the rocBLAS prefill is 241–245 against 235 (+2–4%, about the run-to-run band of the single-die figure) and decode is identical: the "lose or tie" outcome ISA-NOTES §3 expected, not a win, so the guide does not change. The four-die rows were never measured at full clock and will not be: both clamps of the day followed this build's single-die prefill (next section), so the build is retired. One clamped-only observation for the record: at 1000 MHz the rocBLAS tp4 prefill ran 771 tok/s against 538 for MMQ. MMQ's prefill scales almost linearly with sclk (145/235 = 0.62 at 1000/1730 = 0.58), the fp16 GEMM path less so; that says nothing about the healthy box but is consistent with MMQ being compute-bound at the cap.

## Interruption 2: the clamp again at 08:13 UTC, five minutes after the reboot

The box came back at 08:08: fan daemon active, XGMI probe clean twice (every direction 32.8–33.5 GB/s single-flow, SDMA ring 257 GB/s), dies idle at 1730 MHz under perf level high. `cublas-test.sh` started at 08:11:46. From the sampler (`qwen38-27b-q8_0-cublas-clocks.txt.clamped-0813`):

| time | 0b | 0e | 1b | 1e | what was running |
|---|---|---|---|---|---|
| 08:12:32–08:12:57 | 1445–1608 MHz, 196–202 W | 1730, 43 W | 1730, 43 W | 1730, 38 W | FORCE_CUBLAS single-die pp2048 at the cap |
| 08:13:02 | 1730, 40 W | 1730, 38 W | **1000**, 38 W | **1000**, 37 W | idle, between the single-die and four-die tests |
| 08:13:07 | **1000**, 39 W | 1730 | 1730 | 1730 | idle |
| 08:13:12 | 1730 | **1000** | 1730 | **1000** | idle |
| 08:13:17 onward | 1000, 31 W | 1000 | 1000 | 1000 | idle, then ~85 W each under the tp4 test |

The dies dropped while idle and cool, one or two at a time over 15 s, right after die 0b had spent 25 s at its 200 W cap in the rocBLAS prefill; the 06:47 clamp had the same prelude in the same test. The MMQ build spends its prefill at the same cap (the 2026-09-04 run logged 168 five-second samples of die 0b at ≥ 195 W) without tripping anything, so the trigger is specific to what the rocBLAS GEMM does at the cap (it holds a lower clock there, 1445–1608 MHz against 1608–1730 for MMQ, i.e. more current per clock), and whatever it trips is hive-wide. PCIe Emergency Power Reduction is "Not Supported" on these cards, so it is not the PWRBRK# mechanism. Only sclk is affected: mclk 1000, fclk 1166 and socclk 971 are their normal fixed values.

Tried in order, none of it helpful: perf level high (the idle sclk stays at 1000, which makes "perf high at idle reads 1730" a five-second clamp test), power cap 150 W and back to 200 W, and the driver's GPU reset (`cat /sys/kernel/debug/dri/0000:0b:00.0/amdgpu_gpu_recover`, which resets the whole XGMI hive). The reset made things worse: `psp gfx command UNLOAD_TA failed`, then every die hung for 20 s in its VBIOS init table (`atombios stuck in loop ... asic atom init failed`), `GPU reset end with ret = -22`, and the dies now answer `Device or resource busy` to every sysfs read with no GPU visible to the runtime (`gpu-reset-attempt-0820.dmesg.txt`). The latch sits below the driver's reach, in the SMU or on the board, and only a platform reset clears it. Not tried: driver unbind/rebind and a PCIe slot reset, both of which can leave the box needing a power cycle. Also logged at 08:15: a correctable ECC error on the host DIMM at MC0 channel 1 slot 0 (channel 0 had them on 2026-09-03).

The PSU-budget hypothesis (raised afterwards): the wall meter reads close to the 1280 W continuous rating with four dies pegged and the CPU idle, so an SMC power failsafe would explain a hive-wide, driver-immune, reboot-only clamp. The SMC's own budget table is readable from Linux through `applesmc`'s raw key interface (`smc-dump.py`, `smc-keys-clamped-0813.txt`): eight power zones with an envelope, a reading and a fourth value that is 0.0 at idle. DC total 1228 W (reading 207 W idle), CPU 450 W, MPX bay 1 and bay 2 530 W each (their readings are exactly the two GPU rail keys, 50 and 47 W idle), PCIe 300 W, AC input 1562 W. Against the hypothesis: at both clamp moments today one die was at 200 W and three were idle, about 550 W at the wall; the 2026-09-03 clamps after hours of four-die load fit it directly. `smc-log.sh` now records the zone readings, the fourth values and the per-die telemetry flags every 5 s beside the clock sampler, so the next clamp shows what the SMC saw at that second. The discriminating experiment is a 150 W per-die power cap (about 200 W less at the wall) against the known trigger.

Consequences: `cublas-test.sh` is out of the queue; `clamp-watchdog.sh` (kills the chain when a busy die sits at ≤ 1000 MHz for 2 min) runs beside the queue; the queue runs the five-second test before every job. The stock control rows and the FA_ALL_QUANTS rows taken after 08:13 are void (`*.clamped-0813*`).

**Remaining work, queued in `/root/rocm-tests/bench/queue-after-reboot.sh`** (after the next reboot: clamp pre-check, XGMI probe twice, one GPU job at a time, watchdog beside it): item 8 (FA_ALL_QUANTS), item 9 speed rerun with the corrected library path plus the rocprof counters and kernel trace for item 10, the `NCCL_PROTO=LL` serving A/B and the `GGML_CUDA_GRAPH_OPT=1` single-die check, Q4_1 quality and speed, mixed-traffic layout B, and the two-wave MTP-at-depth rerun; about 3.5 hours.

## Interruption 3: the power-draw test after the cold cycle, 09:13 UTC — the clamp is the SMC's DC power envelope

Cold cycle at 09:09 (shut down, unplugged 30 s, the only thing that resets the SMC); the five-second test read 1730 MHz on all four dies at 09:12, t2fanrd active, the four `I2C_NAK` lines at boot are the same as every previous boot. The meter's peak register was cleared, then `power-draw-test.sh` ran 09:13:20–09:20:52 with `smc-log.sh` and the clock sampler beside it (`power-draw-test.{progress,md}`, `power-draw-test-smc.log`, `power-draw-test-clocks.txt`). Each phase was followed by 30 s idle under perf level high with the five-second test.

| phase | what | s | SMC DC total max W | SMC AC est. max W | bay 1 / bay 2 max W | CPU zone max W | max single die W (sampler) | PZ0T | 30 s idle after |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | idle baseline | 60 | 302 | 358 | 105 / 101 | 37 | 44 | 0 | 1730 |
| 1 | rocBLAS (`FORCE_CUBLAS`) single-die pp2048 on rocm0, `-r 4` — the morning's "known trigger" | 64 | 548 | 675 | 296 / 98 | 84 | 223 | 0 | 1730, no clamp (pp2048 245.2 ± 2.0) |
| 2 | stock tp4 pp8192 `-ub 2048 -r 6`, CPU idle (four dies at the cap) | 90 | 1062 | 1353 | 460 / 453 | 88 | 216 | 0 | 1730, no clamp (pp8192 821.9 ± 0.5) |
| 3 | the same tp4 prefill + `openssl speed` on all 56 threads | 143 | **1264** | **1428** | 459 / 451 | 303 | 218 | **59.1** | **1000 on all four: clamped** (pp8192 577.6 ± 119, straddling the clamp) |

The meter's peak register read **1421 W** after the run, against the SMC's AC-input estimate peak of 1428 W (`PZ7G`), so the SMC's AC key is the meter to within 0.5 % at load (it reads about 40 W high at idle). The DC envelope `PZ0E` is 1228 W; Apple's continuous rating is 1280 W.

What the SMC log shows for phase 3, five-second samples: the CPU zone went to 300 W at 09:18:02 with the dies idle (DC 559 W); the prefill started at 09:18:48 and the DC total sat at 1258–1264 W for four samples (bays 456–459 / 448–451 W); at 09:19:03 `PZ0T`, the fourth value of the DC-total zone that had read 0.0 in every sample ever logged, read 59.1; at 09:19:08 both bays were at 206 / 200 W and the sampler had all four dies at 1000 MHz and ~80 W each, and they stayed there through the 30 s idle at perf level high (`PZ0T` back to 0.0, the die status flags `TG*s` 0 throughout, so the `s=1` seen at 08:13 was the failed GPU reset, not the clamp). Bays 3 and 4 never exceeded 460 W of their 530 W envelopes and `PZ3T`/`PZ4T` stayed 0; the CPU zone stayed at 300 of 450 W. The zone that tripped is zone 0, the DC total, about 15–20 s after it crossed 1228 W.

Reading:

- **Four dies at the 200 W cap with the host idle is inside the envelope by 166 W** (1062 W DC, 1353 W AC estimate). Every GPU-only job in the queue runs at or below that.
- **A full CPU load on top of four dies at the cap is 36 W over the envelope, and the SMC clamps the dies within 20 s.** It clamps the GPUs, not the CPU, hive-wide, and latches: perf level high, idle, and a driver reset do not release it.
- **The rocBLAS single-die prefill is not a trigger on a cold boot** (phase 1: die 0b at 223 W, 548 W DC, no clamp). The 08:13 clamp under that same workload came five minutes after a *warm* reboot from the 06:47 clamped state; the warm reboot brought the clocks back but evidently not the SMC, which re-clamped at ~550 W DC. The 06:47 clamp (3.7 h into the chain, single die at the cap, no SMC log yet) is still unexplained by the envelope alone; the 2026-09-03 clamps after hours of four-die load at the meter's ~1280 W fit it directly. So: after any clamp, cold cycle; never warm-reboot out of one.
- `PZ7G` tracks the meter; `PZ0G` against 1228 is the number to watch. `clamp-watchdog.sh` now also kills the chain when the SMC log shows the DC total at ≥ 1228 W in one sample or ≥ 1200 W in two consecutive samples, i.e. before the SMC acts; `queue-after-reboot.sh` takes a full SMC key dump right after its start-up clamp check so the next run has an unclamped baseline to diff against `smc-keys-clamped-0813.txt` and `smc-keys-clamped-0920.txt` (the two clamped dumps differ in 459 keys, all of them sensor/ADC readings; no latch key is identifiable without a clean dump).
- Still open: whether the latch is SMC-side only (a `PZ0T` threshold) or also a GPU firmware state; whether a 150 W per-die cap (about 200 W less DC) makes four dies plus a loaded CPU safe; and what tripped it at 06:47. Anything that loads the CPU beside a four-die prefill — a kernel build, a second llama instance on the CPU, `openssl speed` — is now known to do it.

Decision (09:40 UTC): cap the host, not the GPUs. The W-3275M's RAPL PL1/PL2 were at 413 W; `gpu-test-env.sh` and `queue-after-reboot.sh` now set them to 150 W for every job and put the old value back afterwards, so four dies at the 200 W cap plus a fully loaded host stay near 1123 W. The GPU caps stay at 200 W because the dies are the device under test; a cap sweep for performance per watt is future work (TODO). `smc-log.sh` now also records the filtered zone values `PZ0F/1F/3F/4F`, in case the 06:47 clamp came from a slower filter than the 5 s reading. Verification is in the section below.

The queue was **not** started: the box is clamped again and needs another cold cycle first. After it, the same two commands apply (five-second test, then `queue-after-reboot.sh`); the power-draw test does not need repeating.

## Power cap: throughput, energy, and where the cap stops working (added 2026-09-08 04:40 UTC)

Energy-integrated cap study on the production build (powercap-adaptive.py, one tp4 server at 16 slots, cap switched live, 32-request waves at 16 clients, bay energy from the SMC's two MPX zones at 1 s). 200 to 80 W in 15 W steps down and up, then 65 to 95 W in 5 W steps four times; down/up samples agree within 0.3%.

| goal | cap per die | agg gen tok/s | gen tok/kJ bays | note |
|---|---:|---:|---:|---|
| max throughput | 200 | 87.1 | 105 | reference |
| within 2% | 185 | 86.6 | 108 | +3% per kJ |
| within 5% | 170 | 84.7 | 113 | +8% per kJ |
| within 10% | 140 | 79.4 | 124 | +18% per kJ |
| max per-watt | 85 or below | 60.6 | 146 | DPM floor: -30% throughput, +39% per kJ |

- Below about 85 W the cap is accepted but not met: the dies sit at sclk level 0 (999-1000 MHz, the firmware's minimum GFX clock; mclk/socclk/fclk have no DPM) and draw ~83 W each under this load. 65, 40, 20 and 10 W all give the same point (floor test, two samples each).
- 8-client validation: the floor gives 110 gen tok/kJ at 40.8 tok/s vs 83.5 at 55.5 tok/s for 200 W; 200 W at 16 clients (105) is about as efficient as the floor at 8. Fill slots before lowering caps.
- HBM2 bandwidth vs cap (hbm-bw HIP probe, four dies at once): read 880-892 GB/s and copy 713-718 GB/s at every cap from 200 to 50 W, dies pinned at 999 MHz from 110 W down. No cap costs bandwidth; the streaming floor is ~115 W per die at any cap of 110 W or below; the cap binds from ~140 W for a streaming load.

Files in /root/rocm-tests/bench: qwen38-27b-mxxmfh-powercap-adaptive.md (+ -points.jsonl), qwen38-27b-mxxmfh-powercap-floor.md, hbm-bw-cap-sweep.md; page regenerated by gen-powercap-section.py.

### When a second node pays (added 2026-09-08 05:1x UTC; hyperconverged three-node quorum)

Marginal DC W per added tok/s on one node, 16 clients: 95->110 W 9.4; 110->125 12.0; 125->140 15.8; 140->155 19.8; 155->170 23.7; 170->185 27.8; 185->200 58. A node at the floor with 16 clients: 60.6 tok/s for 575 W DC; box idle 244 W DC (302 W wall, GPUs auto) -> serving increment 5.5 W per tok/s (1.5 kWh per million generated tokens), 7.0 at 8 clients.
- Hyperconverged, three-node quorum (this fleet): every node's capital/idle/bring-up is sunk; spread across all three at the floor first (182 tok/s for 993 W above idle); raise caps in lockstep only past that, stop at ~125 W (226 tok/s); HA sizing: plan the ceiling at two nodes inside the envelope, 150.6 tok/s from two nodes at the 125 W production cap (2 × 75.3; the earlier 165 assumed 155 W, which the fleet rule does not allow on a hyperconverged node — review 2026-09-08).
- Node powered only to serve: one node up to 86 tok/s (cap ~184 W), two nodes at the floor above.
- Node to be bought: one node to its ceiling; $3000/yr at $0.15/kWh = 2300 W-equivalent, ~$1.58/Mtok capital vs $0.40-0.48 electricity.
- Envelope with the host uncapped (CPU zone 303 W measured): DC 1017 W at 140 W caps, 1078 at 155, 1131 at 170, 1181 at 185, 1215 at 200 under serving load; prefill-heavy at 200 W measured 1264 -> clamp. Hyperconverged nodes: floor to 125 W, never higher.
Wall-side caveat: the SMC's PZ7G runs 1.13-1.46x DC depending on load (not a PSU curve); use the physical meter for wall figures. Model: powercap-node-model.json + gen-powercap-section.py in bench.

### Production envelope with the planned hardware (added 2026-09-08 05:3x UTC)

Host uncapped in production (RAPL 150 W was test-only). Additions per node: 4x M.2 NVMe on a PLX PEX8747 x16->4x4 card, 2x Mellanox ConnectX-4 Lx, 2x 3.5" SATA. Budget (DC W, SMC zones measured + datasheets): CPU zone 29 idle / 86 serving / 303 all-core (RAPL allows 413, zone envelope 450); unzoned 42-55; slot zone today 17; NICs 20/20/28; PLX card 10/10/12; NVMe 6/24/34; SATA 10/18/18 (+~50 W spin-up transient); bays 172 idle, by cap under load (pinned fit: 1.093*4*cap+56).
Non-bay total: 222 serving-host / 468 host all-core + additions max / 595 host at RAPL ceiling. DC totals at the cap: 125 W -> 1047 serving (host 303), 1070 pinned, 1198 pinned at RAPL ceiling; 140 W -> 1112 / 1136 / 1263 (over); 155 W -> 1173 / 1202 / 1329.
Production cap = 125 W per die: economics and envelope agree (30 W margin in the absolute worst case, 181 W under serving with the host all-core). 75.3 tok/s per node, 226 for three. Proposed: a DC-total governor lowering caps past ~1150 W (SMC clamps ~20 s after crossing). Not built.
