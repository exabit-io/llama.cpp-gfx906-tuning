# M1 — kernel trace of decode tokens on the tensor split and on one die (2026-09-08)

**Question** (NEXT-STEPS M1): of the 21.5 ms a single-stream decode token costs on four dies, 8 ms would be weight streaming at HBM speed; where do the other 13 ms go — RCCL kernel time, launch gaps, or the small kernels — and is the HIP graph one per token or split around the allreduces?

**Method.** `tools/m1-trace.sh`: production build (`/opt/llama.cpp-mxxm-fh`, fork tile table + both MMVQ patches, `LD_LIBRARY_PATH` set), Qwen3.8-27B Q8_0, `llama-bench -fa 1 -p 0 -n 64`, `NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16`, perf level high, fans max, host RAPL 150 W, the 5 s clock sampler and the SMC log running. Runs: unprofiled references (`-r 3`) on tp4 (`-dev rocm0/rocm1/rocm2/rocm3 -sm tensor`) and on one die; `rocprofv3 --kernel-trace --stats -f csv` on both; a third tp4 run with `--hip-runtime-trace --rccl-trace` added for the API call counts. `tools/m1-analyze.py` splits each die's trace into tokens at the output-projection matvec (the largest-grid `mul_mat_vec_q`, once per token), takes the median over the 55 steady-state tokens of 65, and reports the span, the union of kernel intervals ("busy"), kernel time by category, counts, and gap sizes. Kernel timestamps are the dispatch packets' own, so kernel durations are unperturbed; the tracer's cost lands between tokens (the profiled `llama-bench` rate is lower, but the median token span under tracing, 21.8 ms, equals the unprofiled 21.5 ms, so the in-token timeline is the real one).

## Results

| | tp4, per die | one die |
|---|---:|---:|
| unprofiled token (`-r 3`) | 21.5 ms (46.5 ± 2.6 tok/s, tg64) | 50.9 ms (19.64 ± 0.04) |
| token span under tracing (median) | 21.7 ms | 51.5 ms |
| kernel dispatches per token per die | 1,866 | 1,658 |
| HIP graph launches per die per token | 129 | — |
| RCCL allreduces per token | 128 (two per block; `ncclGroupStart` / 4 × `ncclAllReduce` / `ncclGroupEnd`) | 0 |
| host `hipStreamSynchronize` per token | 16 | — |
| busy (union of kernel intervals) | 20.4 ms (95%) | 51.0 ms (99%) |
| idle inside the token | 1.1–1.3 ms (5%): ~150 gaps under 10 µs = 0.85 ms; 4–6 gaps of 30–300 µs at the host syncs = 0.4 ms | 0.4 ms |
| matrix-vector kernels (`mul_mat_vec_q`, 433) | 11.2 ms (52%) — 6.3 GiB at **604–607 GB/s**, 68% of the 890 GB/s HBM read | 44.7 ms (88%) — 25.4 GiB at **611 GB/s** |
| RCCL kernels (`ncclDevKernel_Generic_4`, 128) | **3.4–3.5 ms (16%)**, 27 µs each | — |
| small kernels, everything else (1,305 / 1,225) | **5.8 ms (27%)** | 6.4 ms (12.5%) |
| of which: `quantize_q8_1` (257) | 1.0 ms | 1.0 ms |
| `rms_norm` (209) + `l2_norm` (96) | 1.5 ms | 1.6 ms |
| residual adds `k_bin_bcast` (176 / 96) | 0.7 ms | 0.4 ms |
| linear-attention chain, 48 blocks × ~9 kernels (`ssm_conv`, `gated_delta_net`, sigmoid, silu, softplus, concat, 2 × `get_rows`, `set_rows`) | 1.7 ms | 2.4 ms |
| flash attention, 16 blocks × 2 kernels | 0.4 ms | 0.4 ms |
| `cpy` (64), `rope` (32) | 0.4 ms | 0.4 ms |

The four dies agree within 1% on every line. Per-die kernel inventories for one token are in `/root/rocm-tests/bench/trace-m1/` (`tp4-kt`, `tp4-api`, `die0-kt`; run `tools/m1-analyze.py DIR PREFIX --show-token 3`).

## What it settles

1. **The HIP graph is split around every allreduce by construction.** `ggml/src/ggml-backend-meta.cpp` cuts the model graph at every partial-sum node and runs each segment as its own `graph_compute` on each die, with an RCCL group call between segments: 129 `hipGraphLaunch` per die per token. But this costs almost nothing: the host stays ahead of the GPU and the die is busy 95% of the token. Capturing RCCL inside a graph (RCCL 2.30.4 here does support capture) is bounded by the 1.1 ms of idle and is not the lever.
2. **The 13 ms is kernel time, in three parts.** 3.6 ms because the single-column matrix-vector kernel streams at 68% of HBM (the same on one die, so the split costs no matvec efficiency); 3.4 ms of RCCL kernels at 27 µs per allreduce of one hidden vector; 5.8 ms of 1,300 small kernels that run back to back at the ~4 µs dispatch floor of GCN. Plus 1.1 ms idle.
3. **The largest lever is the kernel count** (NEXT-STEPS S3, broadened): fuse the activation quantisation into the preceding norm, the residual add into the following norm, and the linear-attention block into one kernel. 3–4 of the 5.8 ms is realistic: +15–20% single-stream on tp4, +8% on one die, and the saving carries into batched decode, where these kernels do not grow with the batch.
4. **Second, the allreduce kernel** (S2): a direct XGMI peer-store allreduce at 5–8 µs instead of 27 returns 2.5–2.8 ms, +13–15%.
5. **Third, batch-1 matvec bandwidth** (S6): 68% → 85% of HBM is worth 2–3 ms on tp4 and 7–8 ms on one die.
6. The 16 host synchronisations and the in-graph dispatch gaps together are the 1.1 ms; not worth a change on their own.

The three add up: the token's floor with all three done is roughly 21.5 − 3.5 − 2.7 − 2.5 ≈ 13 ms, ~75 tok/s single-stream before MTP, against 46.5 today.

## Notes

- The profiled `llama-bench` rates (28 and 37 tok/s on tp4, 19.0 on one die) are the tracer's cost, not the kernels'; use the unprofiled references.
- `llama-bench` takes `-dev rocm0/rocm1/rocm2/rocm3` (slashes) for one four-die test; commas run separate single-device tests. `BENCHMARKS-TODO.md` had commas.
- The `/opt/llama.cpp-*` builds carry no rpath and `/etc/ld.so.conf.d/llama.cpp.conf` names the stock lib directory; the script sets `LD_LIBRARY_PATH` and logs which `libggml-hip` was loaded.
