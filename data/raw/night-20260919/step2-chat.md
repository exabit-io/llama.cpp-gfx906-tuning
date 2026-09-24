# step2-chat   2026-09-19T15:17:51+00:00

Build `/opt/llama.cpp-gfx906-rocm10`, ROCm 10.0, tp4, `-fa on -ctk q8_0 -ctv q8_0 -ngl all -b 2048 -ub 2048`, 200 W/die.
Sessions per **R2.4**: short user turn, 2K-8K generated, 30% of turns carry a tool result,
context grows from the model's own output. Poisson arrivals (§5.1). `bench/chat-client.py`.
`kernel 7.0.0-31-generic; perf: high high high high ; governor performance; fans rpm: 1188/2529/2512/2521`

| tag | clients | turns | reqs | wall s | seed | gen tok | prefill tok | pre:dec | gen tok/s | req/min | TTFT med / p90 s | decode med / p10 / min tok/s | cache miss | max ctx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| step2-chat-chat-8x48K | 12 | 3 | 24 | 2795 | 24K/50s | 97850 | 510305 | 5.22:1 | 35.0 | 0.52 | 32.55 / 37.92 | 7.4 / 6.0 / 5.0 | 15/24 | 43862 |
| step2-chat-chat-4x128K | 6 | 3 | 12 | 3963 | 100K/267s | 46118 | 1317296 | 28.56:1 | 11.6 | 0.18 | 151.02 / 157.60 | 5.5 / 3.7 / 3.6 | 12/12 | 123045 |
| step2-chat-chat-6x64K | 9 | 3 | 18 | 2695 | 40K/85s | 68197 | 477236 | 7.00:1 | 25.3 | 0.40 | 48.97 / 55.05 | 9.3 / 6.9 / 6.8 | 10/18 | 58796 |

Per-run notes:
  - `chat-8x48K`: nextn IGNORED (MTP off); peak VRAM 13.9 13.8 13.8 13.8  GiB/die
  - `chat-4x128K`: nextn IGNORED (MTP off); peak VRAM 16.5 16.5 16.5 16.5  GiB/die
  - `chat-6x64K`: nextn IGNORED (MTP off); peak VRAM 14.1 14.0 14.0 14.0  GiB/die

Peak SMC DC total: 1129 W (envelope 1228 W).

# done 2026-09-19T17:56:53+00:00
