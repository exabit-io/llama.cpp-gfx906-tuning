# ladder-q8kv-150w — capacity ladder, q8_0 KV, 150 W/die   2026-09-19T21:14:18+00:00

Build `/opt/llama.cpp-gfx906-rocm10` (build 11067, commit 1d1361e7a) on ROCm 10.0, tp4 tensor split, `-fa on`,
`-ngl all`, `-b 2048 -ub 2048`, `-ntg 128`, gfx906.env (custom AR + corrected XGMI ring).
Model Qwen3.8-27B-**Q8_0** (model quant, R2.5) — distinct from the q8_0 **KV** type.
`kernel 7.0.0-31-generic; perf: high high high high ; governor performance; fans rpm: 1190/2518/2514/2511`

R3.1 floor is 12 tok/s **per request**. In batched-bench every sequence decodes in
lockstep, so per-slot = aggregate/slots exactly — there is no median/p10 spread here.

| slots | depth | fits | pp t/s | tg t/s agg | **per slot** | R3.1 | KV MiB/die | compute MiB/die | peak VRAM GiB/die | cell s |
|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
