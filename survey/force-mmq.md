patch:            force-mmq | build option GGML_CUDA_FORCE_MMQ=ON | upstream | force MMQ kernels instead of hipBLAS for quantised mat-mul
axis:             multi-user
zero point:       build-faq (FORCE_MMQ off) measured 2026-09-22
recipe:           4x64K and 1x254K | q8_0 K | 125 W/die | --cache-ram 49152 | -ngl all | RCCL + gated custom AR
metric:           decode tok/s/slot and prefill t/s, full precision; improves requires q<0.10 and effect >= +2%
result:           4x64K decode 15.889 -> 15.888, prefill 750.8 -> 750.9 | 1x254K decode 18.595 -> 18.581, prefill 395.1 -> 395.2 (n=4 per arm)
effect:           -0.01% to +0.02% on every metric
stats:            p=0.9143 / 0.8571 / 0.3714 / 0.7429 — nothing approaches significance | n=4 fresh per arm per cell
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  a model or quantisation whose shapes fall on the other side of the heuristic threshold, where forcing could matter
notes:            A clean null on gfx906 with Q8_0 weights: the dispatch heuristic was already choosing MMQ,
                  so forcing it changes nothing. Distinct from GGML_CUDA_FORCE_CUBLAS, a recorded loser
                  -- that is the opposite switch and its verdict says nothing about this one.
