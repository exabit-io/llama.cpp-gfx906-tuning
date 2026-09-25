patch:            fa-head256-rows | https://github.com/exabit-io/mx-llama.cpp/tree/gfx906-candidates | bf4bf4f19 60cad6022 | gfx906 GCN row for the head-256 flash-attention tile table (terms 22 27)
axis:             single-user
zero point:       build-ps-base = master a23e12438 measured 2026-09-24/25 (r1, interleaved in the same blocks)
recipe:           1x255K (1 x 260864 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level median of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode median 25.916 (median 25.916 / p10 25.894 / min 25.889) vs ref 30.916 tok/s | prefill median 398.3 (min 398.0) vs ref 398.5 t/s | n=5 per arm | decode spread 0.34%
effect:           decode -16.17%, prefill -0.07% vs base
stats:            decode p=0.0476 q=0.0741; prefill p=1.0000 q=1.0000 | two-sided exact permutation 5 vs 5, BH over the declared family m=28 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          regresses
bin:              regresses-both
would change if:  the head-256 tile row is re-derived for gfx906 so decode no longer loses 10.7% at 4x64K and 16.2% at 1x255K; prefill is unchanged in both cells, so the loss is in the decode flash-attention path
notes:            Round r1 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user regresses, single-user regresses. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Data: data/raw/night-20260919/binrun.tsv, br-single-fa-head256-rows-*.md, binstats output in the same folder.
