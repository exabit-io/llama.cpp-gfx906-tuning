patch:            v-cache-q5_0 | GGML_CUDA_FA_QUANTS + runtime -ctv | V cache at 5.5 bits per value
axis:             multi-user
zero point:       build-faq-allquants with -ctv q8_0, same binary, one flag differs, measured 2026-09-23
recipe:           4x64K, 8x32K, 1x254K | -ctk q8_0 | 125 W/die | --cache-ram 49152 | -ngl all | RCCL + gated custom AR | FA_QUANTS=all
metric:           decode tok/s/slot, full precision; a drop is actionable at any size because the alternative is free
result:           4x64K -4.77%, 8x32K -7.79%, 1x254K -3.22% against q8_0-V (n=2 per cell)
effect:           -4.77% / -7.79% / -3.22%
stats:            n=2 SCREEN — the exact permutation floor at n=2 vs n=2 is 0.333, so no q value is attainable and none is claimed
evidence:         screened-only
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  a confirmation run at n=4 showed the direction reversing, or a different model's KV/compute balance changed the dequant cost
notes:            SLOWER than q8_0-V on all three cells, despite carrying FEWER bits (5.5 vs 8.5). Consistent in
                  direction and size across three independent shapes, so the screen is sufficient to DROP
                  it: the alternative (q8_0-V) is free and strictly better here. No confirmation run is
                  warranted -- confirming a loser buys nothing.
                  This is what makes the V-width curve NON-MONOTONIC and kills the bandwidth explanation:
                  fewer bits is not faster. 5-bit unpacking (nibble plus a scattered high bit) is simply a
                  worse dequant path on gfx906 than either 8-bit or 4-bit.
