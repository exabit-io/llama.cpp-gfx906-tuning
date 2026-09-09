# Qwen3.8-27B Q8_0 rerun, 2026-09-04 05:26-06:34 UTC

Conditions: fresh boot (03:59 UTC), sclk unclamped (1730 MHz under load), `rocm-smi --setperflevel high`, chassis fans
at max (set by the user; not readable from Linux). Same commands as the 2026-09-03 set: llama-bench sweep (1/2/4 dies,
depths 0/16k/32k, -r 2), llama-batched-bench tensor split 4 dies, server sweeps tp4 / layer4 / tp2 with server-bench.py
(1300-token prompts, 256 generated, concurrency 1..32). Driver: rerun-27b-q8_0-unclamped.sh. Every stage exit 0,
68 minutes.

Files: qwen38-27b-q8_0-unclamped.md (llama-bench), -batched.md, -server-{tp4,layer4,tp2}.md/.json, -clocks.txt
(sclk/junction/power every 5 s), -clocks-by-stage.md, -COMPARE.md (old vs new, from compare-27b-runs.py).

## Result: no measurable change

126 compared cells, median +0.6%, everything within +/-2% except layer4 at concurrency 32 (-7% aggregate gen, +16%
TTFT). That one pair is not like-for-like: the 09-03 figure came from a separate retry on a fresh server after a GPU
page fault, the new one from the end of a full sweep with 32 used slots. This time layer4 ran through concurrency 32
without a fault.

| benchmark | 09-03 | 09-04 |
|---|---|---|
| llama-bench 1 die pp2048 / tg256 | 234.4 / 20.06 t/s | 234.7 / 20.25 |
| llama-bench 2 dies pp2048 / tg256 | 435.5 / 30.97 | 439.5 / 31.23 |
| llama-bench 4 dies pp2048 / tg256 | 823.5 / 45.40 | 829.7 / 45.69 |
| llama-bench 4 dies at depth 32768, pp / tg | 621.5 / 42.64 | 627.3 / 42.88 |
| batched tp4, batch 32, pp / tg / total | 824 / 178.7 / 679.6 | 830 / 179.2 / 684.1 |
| server tp4 agg gen, conc 1 / 8 / 32 | 32.6 / 64.1 / 66.3 | 32.7 / 64.4 / 66.1 |
| server layer4 agg gen, conc 1 / 8 / 32 | 13.4 / 25.8 / 30.3 (retry) | 13.6 / 25.9 / 28.1 |
| server tp2 agg gen, conc 1 / 8 / 32 | 21.5 / 40.2 / 42.1 | 21.8 / 40.5 / 42.5 |

## Why nothing moved

* The 09-03 27B Q8_0 runs were not clamped: the llama-bench and batched runs preceded that boot's clamp onset
  (10:08-10:17 UTC) and the server sweeps ran on the following boot before its onset (14:57). So this is default DPM
  and default fans versus perf level high and max fans, and the clocks were already at 1730 MHz both times.
* The dies are power-limited, not clock-limited: the single-die llama-bench sits at the 200 W cap (instantaneous
  readings up to 226-284 W), so forcing the top DPM level cannot add anything.
* Max fans lowered junction temperatures (81 C max on the Slot-1 dies in single-die work, 62-80 C with all four
  loaded) without changing throughput; nothing was thermally throttling before either.
* 1000 MHz samples in the trace are idle moments (during tp4: dies reading 1000 MHz average 57 W, dies at 1730 average
  150 W). 71% of layer4 samples show an idle die, as expected for pipeline layer split where one die works at a time.

## Batch-size staircase (fine sweep, 2026-09-04 08:4x UTC, batched-fine.sh)

llama-batched-bench with -npl 1,2,4,6,...,30,32 (512-token prompts; 2048-token confirmation in
qwen38-27b-q8_0-batched-fine-pp2048.md). The "dip at 16" is a staircase in the decode-step time:

| batch | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gen t/s, all seqs | 44.9 | 71.9 | 121.9 | 145.5 | 162.4 | 105.4 | 123.9 | 138.2 | 152.6 | 131.8 | 142.8 | 152.7 | 163.4 | 156.5 | 164.6 | 173.6 | 181.4 |
| ms per step | 22 | 28 | 33 | 41 | 49 | 95 | 97 | 101 | 105 | 137 | 140 | 144 | 147 | 166 | 170 | 173 | 176 |

Cause, verified in the llama.cpp b10288 source on the box (ggml/src/ggml-cuda): decode batches of <= 8 tokens use the
quantised mat-vec kernel (MMVQ_MAX_BATCH_SIZE 8, mmvq.cuh); larger batches use MMQ, whose launcher
(mul_mat_q_switch_J, mmq.cuh) loops J = 8, 16, ..., 128 and keeps the first tile width that covers the batch in one
tile, computing the whole tile. So batch 10 costs like 16, 18 like 24, 26 like 32. MMVQ is also the more efficient
kernel on gfx906 (8 tokens: 49 ms/step; 16 via MMQ: ~100 ms).

Rule for serving on this box: keep the number of concurrently decoding sequences at 8 (latency) or 32 (throughput);
9-15 and 17-23 pay a full tile for a partial one. Batch 16 = batch 8 in aggregate at twice the latency.

## Complete batch-size curve, 1..32 (batched-fine2.sh, 2026-09-04 09:5x-10:18 UTC)

PP 512 in ONE run over every size (qwen38-27b-q8_0-batched-fine2-pp512-all.md); PP 2048 odd sizes in
qwen38-27b-q8_0-batched-fine2-pp2048-odd.md (even sizes in qwen38-27b-q8_0-batched-fine-pp2048.md). The first fine
pass's PP 512 file was overwritten by a mis-tagged run and is kept as *.OVERWRITTEN-by-odd-run.md (its numbers survive
in the table above and agree with this run within 1.5%).

| batch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gen t/s, all seqs | 44.2 | 70.6 | 100.6 | 121.1 | 138.1 | 146.6 | 156.3 | 162.1 | 96.6 | 106.7 | 115.4 | 123.9 | 130.8 | 138.2 | 145.6 | 152.7 | 125.9 | 131.7 | 136.5 | 142.5 | 147.4 | 152.5 | 157.8 | 163.1 | 151.6 | 156.5 | 160.8 | 164.6 | 169.3 | 173.7 | 176.9 | 181.5 |
| ms per step | 23 | 28 | 30 | 33 | 36 | 41 | 45 | 49 | 93 | 94 | 95 | 97 | 99 | 101 | 103 | 105 | 135 | 137 | 139 | 140 | 142 | 144 | 146 | 147 | 165 | 166 | 168 | 170 | 171 | 173 | 175 | 176 |

Worst sizes: 9 (96.6 t/s), 17 (125.9), 25 (151.6) = first size of each MMQ tile. Best per tile: 8 (162.1), 16 (152.7),
24 (163.1), 32 (181.5). PP 2048 tracks PP 512 within 1-4% at every size.

## Serving at every concurrency 1..32 (server-fine.sh, 2026-09-04 10:37-16:53 UTC)

One ascending sweep per placement on a fresh server, same flags and client as the six-level run (1300-token prompts,
256 generated, 2x concurrency requests, min 8). Files qwen38-27b-q8_0-serverfine-server-{tp4,tp2,layer4}.md/.json/.log,
clocks in qwen38-27b-q8_0-serverfine-clocks.txt (no clamp: dies at 1730 MHz averaged 140-170 W, max junction 78 C).

| clients | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tp4 agg gen t/s | 32.7 | 45.6 | 51.6 | 59.5 | 46.9 | 49.7 | 43.8 | 67.5 | 48.0 | 50.3 | 52.2 | 52.8 | 52.7 | 55.1 | 44.5 | 51.6 | 50.5 | 54.7 | 55.5 | 54.6 | 55.5 | 58.5 | 51.3 | 56.9 | 54.6 | 58.7 | 58.9 | 53.6 | 56.4 | 57.1 | 60.3 | 67.6 |
| tp4 TTFT s | 1.83 | 3.64 | 4.81 | 5.84 | 6.04 | 6.32 | 7.09 | 6.65 | 7.25 | 7.25 | 7.50 | 7.77 | 7.73 | 8.34 | 7.99 | 7.84 | 8.02 | 8.23 | 8.31 | 8.01 | 8.24 | 8.03 | 8.43 | 7.96 | 7.97 | 8.15 | 8.76 | 8.33 | 8.60 | 8.27 | 7.96 | 7.99 |
| tp2 agg gen t/s | 21.7 | 31.2 | 34.7 | 38.3 | 33.2 | 34.7 | 32.0 | 40.6 | 33.1 | 34.5 | 35.4 | 36.0 | 34.8 | 37.0 | 34.9 | 36.6 | 35.1 | 37.6 | 37.7 | 38.4 | 37.4 | 38.8 | 35.1 | 38.4 | 38.8 | 38.8 | 39.1 | 36.7 | 37.6 | 37.2 | 39.2 | 42.8 |
| tp2 TTFT s | 3.28 | 6.60 | 7.73 | 10.02 | 11.75 | 10.75 | 11.87 | 13.31 | 12.24 | 11.89 | 15.39 | 12.69 | 12.61 | 14.84 | 14.02 | 13.35 | 13.25 | 13.66 | 13.62 | 13.02 | 13.33 | 14.26 | 13.24 | 13.44 | 14.05 | 14.84 | 13.92 | 15.19 | 14.22 | 13.79 | 13.57 | 13.76 |
| layer4 agg gen t/s | 13.6 | 20.8 | 23.6 | 27.2 | 25.4 | 26.2 | 28.4 | 24.8 | 25.4 | 26.0 | 27.2 | 28.0 | 26.0 | 26.4 | 26.3 | 27.2 | 26.6 | 27.8 | 28.0 | 27.0 | 27.9 | 27.1 | 27.0 | 27.4 | 27.8 | 28.1 | 28.1 | 27.4 | 27.8 | 28.3 | 28.3 | 28.1 |
| layer4 TTFT s | 5.87 | 10.19 | 12.26 | 15.71 | 17.12 | 16.29 | 16.56 | 21.21 | 17.15 | 20.94 | 18.82 | 19.05 | 18.46 | 21.71 | 20.54 | 19.89 | 21.27 | 19.25 | 20.25 | 20.72 | 20.31 | 20.31 | 19.45 | 20.33 | 19.99 | 20.60 | 21.26 | 21.30 | 21.42 | 20.47 | 19.83 | 20.00 |

Findings: the tensor-split curves are jagged, not saturating. tp4 peaks at 4 / 8 / 32 clients (59.5 / 67.5 / 67.6 t/s)
and troughs at 5 / 7 / 9 / 15 / 23 (44-48); tp2 peaks at 4 / 8 / 32 (38.3 / 40.6 / 42.8), troughs at 5 / 7 / 9
(32-33). Same level, different run: conc 16 gave 64.0 (tp4) and 41.4 (tp2) in the morning six-level run but 51.6 and
36.6 here; the other shared levels agree within 1-4%. Cause: identical requests run in synchronised waves; whenever a
wave's 1300-token prefill overlaps another wave's decoding, every running sequence advances one token per ~2 s prefill
step; how often the waves collide depends on level and run. layer4 is flat at 25-28 t/s from 4 clients up (pipeline
bound, waves matter less) and ran through conc 32 without the 09-03 page fault. TTFT plateaus: tp4 ~8 s, tp2 ~13-15 s,
layer4 ~20 s from 8 clients up.

## Recommendation (agreed with the user 2026-09-04 17:30 UTC)

Run llama-server with 8 slots (-np 8) on the four-die tensor split and let the HTTP layer queue the rest. Eight clients
already reach the ceiling (67.5 t/s vs 67.6 at 32, same 15.8 req/min) at 13.4 t/s per request instead of 2.9; four
clients keep 26.3 per request at 59.5 total, and the 8.0 t/s between four and eight is ~691,200 tokens/day (397k-691k
across the two runs) for 0.8 s more first-token wait, which is inside the run-to-run spread (6.65 vs 7.00 s at eight).
Beyond eight every added client only slows the others; 2.9 t/s per request at 32 is below reading speed. -np 8 also
keeps every decode step on the mat-vec kernel (section 3 stairs). At 8 slots the box delivers ~5.8 M generated tokens
and ~22,700 requests of this shape per day.
