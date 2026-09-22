# ladder-q8kv-125w — capacity ladder, q8_0 KV, 125 W/die   2026-09-19T20:34:43+00:00

Build `/opt/llama.cpp-gfx906-rocm10` (build 11067, commit 1d1361e7a) on ROCm 10.0, tp4 tensor split, `-fa on`,
`-ngl all`, `-b 2048 -ub 2048`, `-ntg 128`, gfx906.env (custom AR + corrected XGMI ring).
Model Qwen3.8-27B-**Q8_0** (model quant, R2.5) — distinct from the q8_0 **KV** type.
`kernel 7.0.0-31-generic; perf: high high high high ; governor performance; fans rpm: 1190/2522/2510/2516`

R3.1 floor is 12 tok/s **per request**. In batched-bench every sequence decodes in
lockstep, so per-slot = aggregate/slots exactly — there is no median/p10 spread here.

| slots | depth | fits | pp t/s | tg t/s agg | **per slot** | R3.1 | KV MiB/die | compute MiB/die | peak VRAM GiB/die | cell s |
|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| 4 | 64K | yes | 757.83 | 65.81 | **16.45** | **PASS** | ? | ? | 12.8 12.8 12.8 12.8  | 367 |
| 6 | 64K | yes | 763.56 | 70.26 | **11.71** | FAIL | ? | ? | 13.9 14.0 13.9 13.9  | 538 |
| 8 | 48K | yes | 827.12 | 85.35 | **10.67** | FAIL | ? | ? | 13.8 13.8 13.7 13.7  | 499 |
| 4 | 128K | yes | 581.30 | 43.67 | **10.92** | FAIL | ? | ? | 16.3 16.3 16.2 16.2  | 926 |

Peak SMC DC total during the ladder:  W (envelope 1228 W).

Layer offload per cell (all must read all-on-GPU):

# done 2026-09-19T21:13:33+00:00
