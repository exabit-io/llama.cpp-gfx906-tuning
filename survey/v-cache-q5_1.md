patch:            v-cache-q5_1 | GGML_CUDA_FA_QUANTS + runtime -ctv | V cache at 6.0 bits per value
axis:             multi-user
zero point:       build-faq-allquants with -ctv q8_0, same binary, one flag differs, measured 2026-09-23
recipe:           4x64K, 8x32K, 1x254K | -ctk q8_0 | 125 W/die | --cache-ram 49152 | -ngl all | RCCL + gated custom AR | FA_QUANTS=all
metric:           decode tok/s/slot, full precision; a drop is actionable at any size because the alternative is free
result:           4x64K -1.21%, 8x32K -4.34%, 1x254K +1.54% against q8_0-V (n=2 per cell)
effect:           -1.21% / -4.34% / +1.54%
stats:            n=2 SCREEN — the exact permutation floor at n=2 vs n=2 is 0.333, so no q value is attainable and none is claimed
evidence:         screened-only
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  a confirmation run at n=4 showed the direction reversing, or a different model's KV/compute balance changed the dequant cost
notes:            Slower than q8_0-V at 4x64K and 8x32K, marginally faster at 1x254K, and never above the 2%
                  materiality floor in the helpful direction. Same conclusion as q5_0 for the same reason:
                  the 5-bit family's dequant path is the problem, not the byte count.
