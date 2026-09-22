# ladder-q8kv-200w — capacity ladder, q8_0 KV, 200 W/die   2026-09-19T12:00:03+00:00

Build `/opt/llama.cpp-gfx906-rocm10` (build 11067, commit 1d1361e7a) on ROCm 10.0, tp4 tensor split, `-fa on`,
`-ngl all`, `-b 2048 -ub 2048`, `-ntg 128`, gfx906.env (custom AR + corrected XGMI ring).
Model Qwen3.8-27B-**Q8_0** (model quant, R2.5) — distinct from the q8_0 **KV** type.
`kernel 7.0.0-31-generic; perf: high high high high ; governor performance; fans rpm: 1191/2517/2504/2505`

R3.1 floor is 12 tok/s **per request**. In batched-bench every sequence decodes in
lockstep, so per-slot = aggregate/slots exactly — there is no median/p10 spread here.

| slots | depth | fits | pp t/s | tg t/s agg | **per slot** | R3.1 | KV MiB/die | compute MiB/die | peak VRAM GiB/die | cell s |
|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| 4 | 64K | yes | 954.95 | 71.01 | **17.75** | **PASS** | ? | ? | 12.8 12.8 12.8 12.8  | 309 |
| 8 | 192K | yes | 593.43 | 41.20 | **5.15** | FAIL | ? | ? | 29.9 29.9 29.9 29.9  | 2690 |
| 4 | 256K | yes | 499.20 | 30.25 | **7.56** | FAIL | ? | ? | 24.5 24.5 24.5 24.5  | 2131 |
| 8 | 64K | yes | 955.31 | 84.69 | **10.59** | FAIL | ? | ? | 15.4 15.5 15.4 15.4  | 572 |
| 4 | 128K | yes | 727.51 | 49.59 | **12.40** | **PASS** | ? | ? | 16.3 16.3 16.2 16.2  | 743 |
| 8 | 96K | yes | 826.40 | 67.17 | **8.40** | FAIL | ? | ? | 19.0 19.1 19.0 19.0  | 978 |
| 8 | 128K | yes | 727.99 | 55.39 | **6.92** | FAIL | ? | ? | 22.7 22.7 22.7 22.7  | 1471 |
| 4 | 192K | yes | 593.26 | 37.70 | **9.43** | FAIL | ? | ? | 20.4 20.4 20.4 20.4  | 1351 |

Peak SMC DC total during the ladder: 1144 W (envelope 1228 W).

Layer offload per cell (all must read all-on-GPU):

# done 2026-09-19T14:50:48+00:00
