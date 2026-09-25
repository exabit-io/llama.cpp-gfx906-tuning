patch:            gdn-producer-fold | https://github.com/exabit-io/mx-llama.cpp/tree/gfx906-candidates | 1d5918cce 91fdc0277 8a4a9b616 8c99f4af7 c7ee50145 bf65d955c d35a9c793 | fold the gated-delta-net producers and the q/k L2 norms into the fused add-norm (terms 04 07 08 10 12 (on top of 02 03))
axis:             multi-user
zero point:       build-ps-norm-add-fusion = master a23e12438 + norm-add-fusion measured 2026-09-24/25 (r1, interleaved in the same blocks)
recipe:           4x64K (4 x 65536 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level mean of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode mean 22.794 (median 22.801 / p10 22.780 / min 22.770) vs ref 22.594 tok/s | prefill mean 760.7 (min 760.3) vs ref 764.3 t/s | n=5 per arm | decode spread 0.14%
effect:           decode +0.88%, prefill -0.47% vs norm-add-fusion
stats:            decode p=0.0079 q=0.0247; prefill p=0.0079 q=0.0247 | two-sided exact permutation 5 vs 5, BH over the declared family m=28 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  its increment over norm-add-fusion (+0.88% decode at 4x64K, +1.18% at 1x255K) grows past +2% on a later substrate; it cannot ship without norm-add-fusion (terms 02 03), which is neutral itself
notes:            Round r1 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text different vs base).
                  The text was read: identical to base up to the last clause ("shorter wavelengths like blue and violet are"
                  vs "(blue and violet)"), coherent — a greedy-decode near-tie flipped by the changed summation order, not breakage.
                  Data: data/raw/night-20260919/binrun.tsv, br-multi-gdn-producer-fold-*.md, binstats output in the same folder.
