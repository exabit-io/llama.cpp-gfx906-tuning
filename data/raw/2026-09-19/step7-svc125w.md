# step7-svc125w   2026-09-19T21:17:01+00:00

Build `/opt/llama.cpp-gfx906-rocm10`, ROCm 10.0, tp4, `-fa on -ctk q8_0 -ctv q8_0 -ngl all -b 2048 -ub 2048`, 125 W/die.
Sessions per **R2.4**: short user turn, 2K-8K generated, 30% of turns carry a tool result,
context grows from the model's own output. Poisson arrivals (§5.1). `bench/chat-client.py`.
`kernel 7.0.0-31-generic; perf: high high high high ; governor performance; fans rpm: 1189/2524/2525/2518`

| tag | clients | turns | reqs | wall s | seed | gen tok | prefill tok | pre:dec | gen tok/s | req/min | TTFT med / p90 s | decode med / p10 / min tok/s | cache miss | max ctx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| step7-svc125w-svc125-4x64K | 6 | 3 | 12 | 1403 | 40K/104s | 44373 | 6307 | 0.14:1 | 31.6 | 0.51 | 1.63 / 1.92 | 18.3 / 17.1 / 17.0 | 1/12 | 58796 |
| step7-svc125w-svc125-6x64K | 9 | 3 | 18 | 2354 | 40K/104s | 68197 | 10017 | 0.15:1 | 29.0 | 0.46 | 1.74 / 2.96 | 11.9 / 9.9 / 8.8 | 1/18 | 58796 |

Per-run notes:
  - `svc125-4x64K`: nextn IGNORED (MTP off); peak VRAM 12.3 12.4 12.3 12.3  GiB/die
  - `svc125-6x64K`: nextn IGNORED (MTP off); peak VRAM 14.1 14.0 14.0 13.9  GiB/die

Peak SMC DC total:  W (envelope 1228 W).

# done 2026-09-19T22:20:40+00:00
