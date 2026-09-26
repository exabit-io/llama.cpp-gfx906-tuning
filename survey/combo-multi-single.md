patch:            combo-multi | https://github.com/exabit-io/mx-llama.cpp/tree/gfx906-candidates | 1d5918cce 91fdc0277 8a4a9b616 8c99f4af7 c7ee50145 bf65d955c d35a9c793 49c6ca3b3 c339a4087 4d6246ad7 7af0ac291 df4c199b2 3a49b322e | combo-both plus mmvq-q8-fastpath (round 1b) (terms 01 02 03 04 05 07 08 09 10 12 15 23 28 + max-ilp flag)
axis:             single-user
zero point:       build-ps-base = master a23e12438 measured 2026-09-24/25 (r1b, interleaved in the same blocks)
recipe:           1x255K (1 x 260864 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level median of n=6; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode median 31.703 (median 31.703 / p10 31.674 / min 31.667) vs ref 30.909 tok/s | prefill median 401.8 (min 401.6) vs ref 398.0 t/s | n=6 per arm | decode spread 0.17%
effect:           decode +2.57%, prefill +0.96% vs base
stats:            decode p=0.0130 q=0.0130; prefill p=0.0130 q=0.0130 | two-sided exact permutation 6 vs 6, BH over the declared family m=8 | n=6 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          improves
bin:              both
would change if:  a re-measure on another kernel or substrate drops either axis' decode gain (+2.53% 4x64K, +2.57% 1x255K) below +2%; on decode the stack is sub-additive (+2.5% measured vs +4.5% predicted from round 1), and mmvq's single-user cost (-1.19% alone) does not appear inside it
notes:            Round r1b of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user improves, single-user improves. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Pushed: master = gfx906-single = gfx906-multi = 82868aa3b (tag gfx906/v0.5.0/r1b/master): the 13 commits
                  + GGML_HIP_GFX906_MAX_ILP (CMake default ON for gfx906), verified byte-identical to the measured arm source.
                  Data: data/raw/night-20260919/binrun-r1b.tsv, br-r1b-single-combo-multi-*.md, binstats output in the same folder.
