patch:            s1b-repacked-matvec | https://github.com/exabit-io/mx-llama.cpp/tree/gfx906-candidates | fd1c2e743 cc46c2333 f9d9662c4 | repacked Q8_0 narrow-batch mat-vec with launch bounds, keeps 4 waves (terms 17 18 19)
axis:             single-user
zero point:       build-ps-base = master a23e12438 measured 2026-09-24/25 (r1, interleaved in the same blocks)
recipe:           1x255K (1 x 260864 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level median of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode median 30.868 (median 30.868 / p10 30.858 / min 30.855) vs ref 30.916 tok/s | prefill median 398.3 (min 398.2) vs ref 398.5 t/s | n=5 per arm | decode spread 0.11%
effect:           decode -0.15%, prefill -0.05% vs base
stats:            decode p=0.0476 q=0.0741; prefill p=0.7143 q=0.7692 | two-sided exact permutation 5 vs 5, BH over the declared family m=28 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  the instrument gains a cell with 9-32 rows per mat-vec, the batch width this patch targets; 4x64K and 1x255K run 4 and 1 rows, where it is neutral (+0.19% / -0.15% decode)
notes:            Round r1 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Data: data/raw/night-20260919/binrun.tsv, br-single-s1b-repacked-matvec-*.md, binstats output in the same folder.
