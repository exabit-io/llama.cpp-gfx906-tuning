patch:            s1b-repacked-matvec | https://github.com/exabit-io/mx-llama.cpp/tree/gfx906-candidates | fd1c2e743 cc46c2333 f9d9662c4 | repacked Q8_0 narrow-batch mat-vec with launch bounds, keeps 4 waves (terms 17 18 19)
axis:             multi-user
zero point:       build-ps-base = master a23e12438 measured 2026-09-24/25 (r1, interleaved in the same blocks)
recipe:           4x64K (4 x 65536 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level mean of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode mean 22.511 (median 22.492 / p10 22.482 / min 22.477) vs ref 22.468 tok/s | prefill mean 760.9 (min 760.1) vs ref 760.9 t/s | n=5 per arm | decode spread 0.32%
effect:           decode +0.19%, prefill -0.00% vs base
stats:            decode p=0.1111 q=0.1556; prefill p=0.9762 q=1.0000 | two-sided exact permutation 5 vs 5, BH over the declared family m=28 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  the instrument gains a cell with 9-32 rows per mat-vec, the batch width this patch targets; 4x64K and 1x255K run 4 and 1 rows, where it is neutral (+0.19% / -0.15% decode)
notes:            Round r1 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Data: data/raw/night-20260919/binrun.tsv, br-multi-s1b-repacked-matvec-*.md, binstats output in the same folder.
