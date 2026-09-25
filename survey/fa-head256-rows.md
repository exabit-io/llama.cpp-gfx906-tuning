patch:            fa-head256-rows | https://github.com/exabit-io/mx-llama.cpp/tree/gfx906-candidates | bf4bf4f19 60cad6022 | gfx906 GCN row for the head-256 flash-attention tile table (terms 22 27)
axis:             multi-user
zero point:       build-ps-base = master a23e12438 measured 2026-09-24/25 (r1, interleaved in the same blocks)
recipe:           4x64K (4 x 65536 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level mean of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode mean 20.072 (median 20.060 / p10 20.038 / min 20.033) vs ref 22.468 tok/s | prefill mean 760.7 (min 760.2) vs ref 760.9 t/s | n=5 per arm | decode spread 0.44%
effect:           decode -10.66%, prefill -0.02% vs base
stats:            decode p=0.0079 q=0.0247; prefill p=0.5952 q=0.6944 | two-sided exact permutation 5 vs 5, BH over the declared family m=28 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          regresses
bin:              regresses-both
would change if:  the head-256 tile row is re-derived for gfx906 so decode no longer loses 10.7% at 4x64K and 16.2% at 1x255K; prefill is unchanged in both cells, so the loss is in the decode flash-attention path
notes:            Round r1 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user regresses, single-user regresses. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Data: data/raw/night-20260919/binrun.tsv, br-multi-fa-head256-rows-*.md, binstats output in the same folder.
