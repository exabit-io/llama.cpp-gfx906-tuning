patch:            dpp-warp-reductions | https://github.com/exabit-io/mx-llama.cpp/tree/gfx906-candidates | 49c6ca3b3 c339a4087 | DPP-based warp reductions on GCN, generic reductions kept where DPP does not apply (terms 23 28)
axis:             multi-user
zero point:       build-ps-base = master a23e12438 measured 2026-09-24/25 (r1, interleaved in the same blocks)
recipe:           4x64K (4 x 65536 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level mean of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode mean 22.784 (median 22.788 / p10 22.766 / min 22.765) vs ref 22.468 tok/s | prefill mean 761.4 (min 760.8) vs ref 760.9 t/s | n=5 per arm | decode spread 0.16%
effect:           decode +1.41%, prefill +0.07% vs base
stats:            decode p=0.0079 q=0.0247; prefill p=0.0952 q=0.1404 | two-sided exact permutation 5 vs 5, BH over the declared family m=28 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  its +1.41% decode at 4x64K (q=0.025; +0.49% at 1x255K) grows past +2% on a later substrate
notes:            Round r1 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Data: data/raw/night-20260919/binrun.tsv, br-multi-dpp-warp-reductions-*.md, binstats output in the same folder.
