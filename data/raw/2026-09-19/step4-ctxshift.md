# step4-ctxshift   2026-09-19T19:51:07+00:00

Build `/opt/llama.cpp-gfx906-rocm10`, ROCm 10.0, tp4, `-fa on -ctk q8_0 -ctv q8_0 -ngl all -b 2048 -ub 2048`, 200 W/die.
Sessions per **R2.4**: short user turn, 2K-8K generated, 30% of turns carry a tool result,
context grows from the model's own output. Poisson arrivals (§5.1). `bench/chat-client.py`.
`kernel 7.0.0-31-generic; perf: high high high high ; governor performance; fans rpm: 1187/2528/2528/2519`

| tag | clients | turns | reqs | wall s | seed | gen tok | prefill tok | pre:dec | gen tok/s | req/min | TTFT med / p90 s | decode med / p10 / min tok/s | cache miss | max ctx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| step4-ctxshift-ctxshift-off | 8 | 4 | 4 | 768 | 26K/134s | 5951 | 9120 | 1.53:1 | 7.7 | 0.31 | 2.69 / 6.51 | 11.6 / 7.3 / 7.3 | 1/4 | 34884 |
| step4-ctxshift-ctxshift-on | 8 | 4 | 4 | 769 | 26K/132s | 5942 | 9129 | 1.54:1 | 7.7 | 0.31 | 2.69 / 6.54 | 11.5 / 7.2 / 7.2 | 1/4 | 34884 |
| step4-ctxshift-ctxshift-on-nockpt | 8 | 4 | 4 | 777 | 26K/131s | 6015 | 35843 | 5.96:1 | 7.7 | 0.31 | 2.16 / 31.03 | 11.6 / 6.3 / 6.3 | 1/4 | 34884 |

Per-run notes:
  - `ctxshift-off`: nextn IGNORED (MTP off); peak VRAM 12.1 12.0 12.0 12.0  GiB/die
  - `ctxshift-on`: nextn IGNORED (MTP off); peak VRAM 12.1 12.0 12.0 12.0  GiB/die
  - `ctxshift-on-nockpt`: nextn IGNORED (MTP off); peak VRAM 12.1 12.0 11.9 11.9  GiB/die

Peak SMC DC total: 1125 W (envelope 1228 W).

# done 2026-09-19T20:31:12+00:00
