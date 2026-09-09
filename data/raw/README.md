# data/raw — every result table, client JSON and clock sample behind the reports

One directory per measurement day, copied from the bench directory on the box (`/root/rocm-tests/bench`).
The `*.md` files are the tables the runner scripts write (`llama-bench` `-lb`, `llama-batched-bench` `-bb`,
server-client summaries); the `*.json` files are the per-request records of the server clients
(`tools/server-bench.py`, `dual-server-bench.py`, `mtp-depth-client.py`, `cold-start-client.py`); the
`*-clocks.txt` files are the per-second sclk/power samples of `tools/gpu-test-env.sh start_sampler`.
`data/benchmarks.json` is the transcription the optimiser reads; every key there names its source table.

| Day | Files | What was measured | Report |
|---|---|---|---|
| 2026-09-03 | 38 | first 27B Q8_0 sweeps on stock b10288: batched staircase at pp512/pp2048, server layer4/tp2/tp4, 9B and MoE comparisons, the `ab-*` A/B client runs | `reports/qwen38-27b-q8_0-gfx906.html` |
| 2026-09-04 | 21 | unclamped rerun after the cold power cycle (batched, servers, clocks by stage, SUMMARY/COMPARE), first serving fine sweeps | `reports/qwen38-27b-q8_0-gfx906.html` s.6, `reports/gfx906-xgmi-ring.html` |
| 2026-09-06 | 70 | the context study: f16/q8_0 ladders to 256K at 1/4/8 slots, dp4 per die, pooled cache, MTP at depth, mixed waves, perplexity of the cache types | `reports/qwen38-27b-ctx-gfx906.html` |
| 2026-09-07 | 104 | the TODO run-through: nineteen environment knobs, 2 × tp2, quant quality, MTP depth sweep, MMVQ variants 1–6, the 16-column patch, cuBLAS/FA-quant builds, the mxxm+fh production candidate, power-cap studies (adaptive, floor, phases), the three routed-proxy mixed runs (`A:`, `B (rerun):`, `control:` JSON) | `reports/2026-09-07-todo-runthrough.html` |
| 2026-09-08 | 522 | the next-steps day and the night shift: M1 trace, fork knobs, kernel folds (`nq-*`), allreduce gate, paired CIs, FA counters, server final, governor, cap phases, gfx906-branch validation, ablation, ppl bisect, `post22`–`post28`, S1b candidates, S1 v2 — see `2026-09-08/MANIFEST.md` for the binary table and the repack setting per row | `reports/2026-09-08-*.md/.html` |
| 2026-09-09 | 6 | server-level client JSON of the S1b a3 vs production repeat (`b3-E-*`, `s1b-E-*`) | `reports/2026-09-08-review-followup.md` §5–6 |

Server logs, `.kld` bases, perplexity logs and the rocprof traces stay on the box (several GB).
