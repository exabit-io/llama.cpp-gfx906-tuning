# ladder-q8kv-200w-p2 — capacity ladder, q8_0 KV, 200 W/die   2026-09-19T14:51:33+00:00

Build `/opt/llama.cpp-gfx906-rocm10` (build 11067, commit 1d1361e7a) on ROCm 10.0, tp4 tensor split, `-fa on`,
`-ngl all`, `-b 2048 -ub 2048`, `-ntg 128`, gfx906.env (custom AR + corrected XGMI ring).
Model Qwen3.8-27B-**Q8_0** (model quant, R2.5) — distinct from the q8_0 **KV** type.
`kernel 7.0.0-31-generic; perf: high high high high ; governor performance; fans rpm: 1187/2520/2509/2509`

R3.1 floor is 12 tok/s **per request**. In batched-bench every sequence decodes in
lockstep, so per-slot = aggregate/slots exactly — there is no median/p10 spread here.

| slots | depth | fits | pp t/s | tg t/s agg | **per slot** | R3.1 | KV MiB/die | compute MiB/die | peak VRAM GiB/die | cell s |
|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| 6 | 64K | yes | 955.83 | 79.21 | **13.20** | **PASS** | ? | ? | 13.9 14.0 13.9 13.9  | 432 |
| 8 | 48K | yes | 1035.48 | 97.46 | **12.18** | **PASS** | ? | ? | 13.8 13.8 13.8 13.8  | 402 |
| 10 | 32K | yes | 1132.36 | 117.94 | **11.79** | FAIL | ? | ? | 13.1 13.1 13.1 13.1  | 311 |
| 12 | 32K | yes | 1132.20 | 123.59 | **10.30** | FAIL | ? | ? | 13.7 13.7 13.7 13.7  | 371 |

Peak SMC DC total during the ladder: 1140 W (envelope 1228 W).

Layer offload per cell (all must read all-on-GPU):

# done 2026-09-19T15:16:49+00:00
