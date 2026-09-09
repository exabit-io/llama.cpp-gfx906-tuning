# Draft comment for https://github.com/ggml-org/llama.cpp/pull/27841 (not posted; the user posts it)

Measurements on a 4 × Vega 20 (Radeon Pro Vega II Duo, gfx906, 32 GB HBM2 each, XGMI ring) box under ROCm 7.14 (TheRock, ML-gfx906),
Qwen3.8-27B Q8_0, `-sm tensor` over the four dies and one die alone, hipcc `-mllvm -amdgpu-sched-strategy=max-ilp`.

We carry the same Q8_0 rows this PR proposes (512 threads = 8 wave64, I = 128, tiles to J = 128) in our gfx906 branch
(https://github.com/exabit-io/llama.cpp, from mxxm-t/mx-llama.cpp), and measured them in isolation on upstream master 5d806aa25 with
nothing else changed (Q8_0 only; the RDNA2 table for every other type):

| | four dies pp2048 | one die pp2048 | batched 32 rows | 1/8/16 rows, tg128 | perplexity 16K |
|---|---|---|---|---|---|
| master | 844.6 | 230.3 | — | — | 5.6216 |
| + Q8_0 GCN rows, applied from J >= 32 | +29% (1100.7) | +34% (315.4) | +15% | parity | 5.6216 (identical) |

Two observations that may help with the regressions Johannes measured:

1. Applying the wide rows only from J >= 32 avoids a −7.5% at 16 decode rows we saw when the table applied from J = 8 (the J = 16
   entry); at 1/8/16 rows the RDNA2 row is as fast or faster.
2. The gain is Q8_0-specific in our ablation (the K-quant gains in the fork come from separate K-quant kernels), so landing the Q8_0
   rows first and leaving the other types on the RDNA2 table until each is measured would take the IQ regressions off the table.

Patch of the Q8_0-only variant against master 5d806aa25, and the raw llama-bench / batched-bench tables:
https://github.com/exabit-io/llama.cpp-gfx906-tuning/tree/main/patches/upstream-S1 (`test-backend-ops -o MUL_MAT` 1288/1288).
