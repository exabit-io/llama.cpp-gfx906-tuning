# post22b 2026-09-08T16:26:28+00:00: gfx906 branch (r2) tp4 single-stream instability, knob sweep; llama-bench -p 0 -n 128 -r 3 --no-repack, gfx906.env base
| variant | tp4 tg128 |
|---|---:|
| base | 37.88 ± 17.60 |
| no-graphs | 42.58 ± 15.66 |
| token-graph-off | 39.24 ± 15.26 |
| parallel-dispatch-off | 39.20 ± 15.11 |
| graph-reuse-off | 20.44 ± 5.56 |
| graph-opt-0 | 38.55 ± 16.19 |
| hwqueues-8 | 38.03 ± 16.36 |
| no-dedicated-cpy | 39.85 ± 16.19 |
| no-custom-ar | 37.07 ± 13.17 |
| all-fork-off | 14.03 ± 2.88 |
| base-repeat | 46.46 ± 14.21 |

## bisect anchors: upstream at the fork base 0f3a71be1 and the fork pristine b10912 (ppl 16K/6)
| 0f3a71be1 Final estimate: PPL = 5.6118 +/- 0.06229 |
| llama.cpp-b10912 Final estimate: PPL = 5.6143 +/- 0.06230 |
# done 2026-09-08T16:42:44+00:00
