# step2b-cram   2026-09-19T17:57:52+00:00

Build `/opt/llama.cpp-gfx906-rocm10`, ROCm 10.0, tp4, `-fa on -ctk q8_0 -ctv q8_0 -ngl all -b 2048 -ub 2048`, 200 W/die.
Sessions per **R2.4**: short user turn, 2K-8K generated, 30% of turns carry a tool result,
context grows from the model's own output. Poisson arrivals (§5.1). `bench/chat-client.py`.
`kernel 7.0.0-31-generic; perf: high high high high ; governor performance; fans rpm: 1190/2528/2522/2294`

| tag | clients | turns | reqs | wall s | seed | gen tok | prefill tok | pre:dec | gen tok/s | req/min | TTFT med / p90 s | decode med / p10 / min tok/s | cache miss | max ctx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| step2b-cram-cram-8x48K | 12 | 3 | 24 | 2315 | 24K/50s | 97850 | 14168 | 0.14:1 | 42.3 | 0.62 | 1.36 / 2.84 | 12.1 / 7.1 / 6.5 | 0/24 | 43862 |
| step2b-cram-cram-4x128K | 6 | 3 | 12 | 2224 | 100K/268s | 46118 | 8859 | 0.19:1 | 20.7 | 0.32 | 2.53 / 5.54 | 12.3 / 11.6 / 11.6 | 1/12 | 123045 |
| step2b-cram-cram-6x64K | 9 | 3 | 18 | 2125 | 40K/91s | 68197 | 6456 | 0.09:1 | 32.1 | 0.51 | 1.48 / 1.75 | 13.2 / 10.1 / 9.0 | 0/18 | 58796 |

Per-run notes:
  - `cram-8x48K`: nextn IGNORED (MTP off); peak VRAM 13.9 13.8 13.8 13.8  GiB/die
  - `cram-4x128K`: nextn IGNORED (MTP off); peak VRAM 16.5 16.5 16.5 16.4  GiB/die
  - `cram-6x64K`: nextn IGNORED (MTP off); peak VRAM 14.2 14.1 14.0 14.0  GiB/die

Peak SMC DC total: 1130 W (envelope 1228 W).

# done 2026-09-19T19:50:32+00:00
