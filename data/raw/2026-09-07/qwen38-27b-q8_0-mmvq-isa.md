# MMVQ Q8_0 kernel instruction counts on gfx906 (static, from `clang -x hip --cuda-device-only -S` with the installed build's flags)  2026-09-07T03:20:14+00:00

Assembly: /tmp/claude-0/-/b2fc75c5-25d0-4253-b097-a756ea19f02f/scratchpad/mmvq-{stock,patched}-gfx906.s; counter: /tmp/claude-0/-/b2fc75c5-25d0-4253-b097-a756ea19f02f/scratchpad/isa-count.py. 'main loop' = the largest backward-branch loop in the kernel (the K loop over 32-weight blocks). Each thread handles vdr=2 dwords (8 int8) of one block per column per row per iteration; GCN table: nwarps=2 for 1..4 columns, 1 for 5..8; rows per block 1 (ncols 1) or 2. 'per col x row' = loop instructions / (ncols x rows).

## stock b10288
| kernel | VGPRs | SGPRs | LDS B | occupancy | total inst | main loop: VALU | SALU | SMEM | LDS | VMEM | waitcnt | branch | loop total | dot4 (v_dot4) | per col x row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 ncols=1 | 19 | 28 | ? | ? | 177 | 15 | 2 | 0 | 0 | 4 | 3 | 1 | 25 | 2 | 25.0 |
| Q8_0 ncols=1 small_k | 26 | 31 | ? | ? | 229 | 24 | 2 | 0 | 0 | 6 | 4 | 1 | 37 | 4 | 37.0 |
| Q8_0 ncols=2 | 32 | 35 | ? | ? | 283 | 37 | 2 | 0 | 0 | 8 | 7 | 1 | 55 | 8 | 13.8 |
| Q8_0 ncols=3 | 41 | 40 | ? | ? | 384 | 52 | 3 | 0 | 0 | 10 | 7 | 1 | 73 | 12 | 12.2 |
| Q8_0 ncols=4 | 48 | 40 | ? | ? | 471 | 66 | 3 | 0 | 0 | 12 | 11 | 1 | 93 | 16 | 11.6 |
| Q8_0 ncols=5 | 55 | 44 | ? | ? | 510 | 80 | 3 | 0 | 0 | 14 | 11 | 1 | 109 | 20 | 10.9 |
| Q8_0 ncols=6 | 62 | 44 | ? | ? | 589 | 94 | 3 | 0 | 0 | 16 | 13 | 1 | 127 | 24 | 10.6 |
| Q8_0 ncols=7 | 64 | 44 | ? | ? | 666 | 108 | 3 | 0 | 0 | 18 | 14 | 1 | 144 | 28 | 10.3 |
| Q8_0 ncols=8 | 64 | 48 | ? | ? | 744 | 122 | 3 | 0 | 0 | 20 | 15 | 1 | 161 | 32 | 10.1 |

## patched MMVQ_MAX_BATCH_SIZE=16 (worktree /root/llama.cpp-mmvq16, same tables; 9..16 columns get rows=2, nwarps=1)
| kernel | VGPRs | SGPRs | LDS B | occupancy | total inst | main loop: VALU | SALU | SMEM | LDS | VMEM | waitcnt | branch | loop total | dot4 (v_dot4) | per col x row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 ncols=9 | 70 | 49 | ? | ? | 816 | 136 | 3 | 0 | 0 | 22 | 11 | 1 | 173 | 36 | 9.6 |
| Q8_0 ncols=10 | 77 | 53 | ? | ? | 900 | 150 | 3 | 0 | 0 | 24 | 18 | 1 | 196 | 40 | 9.8 |
| Q8_0 ncols=11 | 83 | 55 | ? | ? | 971 | 164 | 3 | 0 | 0 | 26 | 13 | 1 | 207 | 44 | 9.4 |
| Q8_0 ncols=12 | 89 | 59 | ? | ? | 1049 | 178 | 3 | 0 | 0 | 28 | 14 | 1 | 224 | 48 | 9.3 |
| Q8_0 ncols=13 | 95 | 61 | ? | ? | 1126 | 192 | 3 | 0 | 0 | 30 | 15 | 1 | 241 | 52 | 9.3 |
| Q8_0 ncols=14 | 101 | 65 | ? | ? | 1204 | 206 | 3 | 0 | 0 | 32 | 16 | 1 | 258 | 56 | 9.2 |
| Q8_0 ncols=15 | 107 | 68 | ? | ? | 1281 | 220 | 3 | 0 | 0 | 34 | 17 | 1 | 275 | 60 | 9.2 |
| Q8_0 ncols=16 | 113 | 70 | ? | ? | 1359 | 234 | 3 | 0 | 0 | 36 | 18 | 1 | 292 | 64 | 9.1 |
