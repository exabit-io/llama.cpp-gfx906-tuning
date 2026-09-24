# step9-prodfans   2026-09-19T22:21:34+00:00

Build `/opt/llama.cpp-gfx906-rocm10`, ROCm 10.0, tp4, `-fa on -ctk q8_0 -ctv q8_0 -ngl all -b 2048 -ub 2048`, 125 W/die.
Sessions per **R2.4**: short user turn, 2K-8K generated, 30% of turns carry a tool result,
context grows from the model's own output. Poisson arrivals (§5.1). `bench/chat-client.py`.
`kernel 7.0.0-31-generic; perf: high high high high ; governor performance; fans rpm: 503/485/488/487`

| tag | clients | turns | reqs | wall s | seed | gen tok | prefill tok | pre:dec | gen tok/s | req/min | TTFT med / p90 s | decode med / p10 / min tok/s | cache miss | max ctx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| step9-prodfans-pf125-4x64K | 6 | 3 | 12 | 1411 | 40K/105s | 44373 | 6307 | 0.14:1 | 31.5 | 0.51 | 1.75 / 1.97 | 18.2 / 17.0 / 17.0 | 1/12 | 58796 |

Per-run notes:
  - `pf125-4x64K`: nextn IGNORED (MTP off); peak VRAM 12.3 12.4 12.3 12.3  GiB/die

Peak SMC DC total:  W (envelope 1228 W).

# done 2026-09-19T22:45:36+00:00
