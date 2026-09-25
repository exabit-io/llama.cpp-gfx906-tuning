patch:            norm-add-fusion | https://github.com/exabit-io/mx-llama.cpp/tree/gfx906-candidates | 1d5918cce 91fdc0277 | the fused RMS_NORM+MUL+ADD also emits the Q8_1 activations and computes the residual ADD inside the fusion (terms 02 03)
axis:             multi-user
zero point:       build-ps-base = master a23e12438 measured 2026-09-24/25 (r1, interleaved in the same blocks)
recipe:           4x64K (4 x 65536 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level mean of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode mean 22.594 (median 22.589 / p10 22.576 / min 22.571) vs ref 22.468 tok/s | prefill mean 764.3 (min 763.7) vs ref 760.9 t/s | n=5 per arm | decode spread 0.22%
effect:           decode +0.56%, prefill +0.45% vs base
stats:            decode p=0.0079 q=0.0247; prefill p=0.0079 q=0.0247 | two-sided exact permutation 5 vs 5, BH over the declared family m=28 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  a later substrate makes the fused add-norm a larger share of decode: today +0.56% decode / +0.45% prefill at 4x64K (both q=0.025) and +0.08% / +0.14% at 1x255K, real but below the 2% floor
notes:            Round r1 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Data: data/raw/night-20260919/binrun.tsv, br-multi-norm-add-fusion-*.md, binstats output in the same folder.
