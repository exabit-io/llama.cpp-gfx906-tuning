patch:            norm-add-fusion | https://github.com/exabit-io/mx-llama.cpp/tree/gfx906-candidates | 1d5918cce 91fdc0277 | the fused RMS_NORM+MUL+ADD also emits the Q8_1 activations and computes the residual ADD inside the fusion (terms 02 03)
axis:             single-user
zero point:       build-ps-base = master a23e12438 measured 2026-09-24/25 (r1, interleaved in the same blocks)
recipe:           1x255K (1 x 260864 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level median of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode median 30.941 (median 30.941 / p10 30.915 / min 30.914) vs ref 30.916 tok/s | prefill median 399.1 (min 399.0) vs ref 398.5 t/s | n=5 per arm | decode spread 0.26%
effect:           decode +0.08%, prefill +0.14% vs base
stats:            decode p=0.1667 q=0.2222; prefill p=0.0476 q=0.0741 | two-sided exact permutation 5 vs 5, BH over the declared family m=28 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  a later substrate makes the fused add-norm a larger share of decode: today +0.56% decode / +0.45% prefill at 4x64K (both q=0.025) and +0.08% / +0.14% at 1x255K, real but below the 2% floor
notes:            Round r1 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Data: data/raw/night-20260919/binrun.tsv, br-single-norm-add-fusion-*.md, binstats output in the same folder.
