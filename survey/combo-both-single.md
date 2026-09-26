patch:            combo-both | https://github.com/exabit-io/mx-llama.cpp/tree/gfx906-candidates | 1d5918cce 91fdc0277 8a4a9b616 8c99f4af7 c7ee50145 bf65d955c d35a9c793 49c6ca3b3 c339a4087 | stack of norm-add-fusion, gdn-producer-fold, dpp-warp-reductions and max-ilp (round 1b) (terms 02 03 04 07 08 10 12 23 28 + max-ilp flag)
axis:             single-user
zero point:       build-ps-base = master a23e12438 measured 2026-09-24/25 (r1b, interleaved in the same blocks)
recipe:           1x255K (1 x 260864 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level median of n=6; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode median 31.617 (median 31.617 / p10 31.596 / min 31.586) vs ref 30.909 tok/s | prefill median 401.7 (min 401.6) vs ref 398.0 t/s | n=6 per arm | decode spread 0.24%
effect:           decode +2.29%, prefill +0.93% vs base
stats:            decode p=0.0130 q=0.0130; prefill p=0.0130 q=0.0130 | two-sided exact permutation 6 vs 6, BH over the declared family m=8 | n=6 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          improves
bin:              single-user-only
would change if:  a later substrate moves its multi-user decode (+1.95%, 0.05 pt under the floor) past +2%; combo-multi, which contains it, binned both and supersedes it
notes:            Round r1b of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user neutral, single-user improves. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Superseded by combo-multi (a superset that beat it on both axes and binned both); not pushed on its own.
                  Data: data/raw/night-20260919/binrun-r1b.tsv, br-r1b-single-combo-both-*.md, binstats output in the same folder.
