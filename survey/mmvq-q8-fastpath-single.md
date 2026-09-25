patch:            mmvq-q8-fastpath | https://github.com/exabit-io/mx-llama.cpp/tree/gfx906-candidates | 4d6246ad7 7af0ac291 df4c199b2 3a49b322e | gfx906 MMVQ: 16-column width (MMID stays 8), batch-1 rows/warps-per-block knobs, whole-block vdr-8 load at one column, MUL_MAT_ID sync fix (terms 01 05 09 15, one unit)
axis:             single-user
zero point:       build-ps-base = master a23e12438 measured 2026-09-24/25 (r1, interleaved in the same blocks)
recipe:           1x255K (1 x 260864 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level median of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode median 30.549 (median 30.549 / p10 30.544 / min 30.541) vs ref 30.916 tok/s | prefill median 398.6 (min 397.9) vs ref 398.5 t/s | n=5 per arm | decode spread 0.14%
effect:           decode -1.19%, prefill +0.02% vs base
stats:            decode p=0.0476 q=0.0741; prefill p=0.7143 q=0.7692 | two-sided exact permutation 5 vs 5, BH over the declared family m=28 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  the batch-1 knobs are re-tuned for long context: they cost 1.19% decode at 1x255K (q=0.074) and gain 0.57% at 4x64K (q=0.025), both inside the 2% floor
notes:            Round r1 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Data: data/raw/night-20260919/binrun.tsv, br-single-mmvq-batch1-knobs-*.md, binstats output in the same folder.
