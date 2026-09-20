patch:            tech-gemv-gemm-threshold | https://gitee.com/bjxamo_admin/vllm-gfx906 | n/a (vLLM fork, not llama.cpp) | quantization GEMM-vs-GEMV switchover threshold lowered from 50/24 to 8/8 for gfx906
axis:             multi-user
zero point:       not applicable until implemented — nothing to A/B against yet
recipe:           4x64K | q8_0 KV | 125 W/die | --cache-ram 49152 | -ngl all | settings/gfx906.env
metric:           per-request decode tok/s at the design point; adopted if >= 16.45 + noise band
result:           median n/a / p10 n/a / min n/a tok/s | n=0 | spread n/a
effect:           n/a — not implemented here
stats:            n/a — no confirmation runs collected
evidence:         inspection
structural:       standalone
verdict:          untested
bin:              technique-requires-implementation
would change if:  a sweep of llama.cpp's own MMVQ/MMQ switchover shows the current threshold is
                  already optimal on gfx906, in which case this closes as neutral.
notes:            Independent corroboration, not a portable patch: a different engine converged on a
                  threshold of 8 for gfx906, and our own batch-size staircase measured MMVQ in use at
                  <=8 with decode-cost steps at 8/16/24. Two engines agreeing makes the switchover
                  worth an explicit sweep rather than an assumption. Requires implementing the sweep
                  in llama.cpp before it can be surveyed at all — hence the bin.
