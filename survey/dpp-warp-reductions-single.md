patch:            dpp-warp-reductions | https://github.com/exabit-io/mx-llama.cpp/tree/gfx906-candidates | 49c6ca3b3 c339a4087 | DPP-based warp reductions on GCN, generic reductions kept where DPP does not apply (terms 23 28)
axis:             single-user
zero point:       build-ps-base = master a23e12438 measured 2026-09-24/25 (r1, interleaved in the same blocks)
recipe:           1x255K (1 x 260864 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level median of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode median 31.066 (median 31.066 / p10 31.039 / min 31.023) vs ref 30.916 tok/s | prefill median 398.7 (min 398.0) vs ref 398.5 t/s | n=5 per arm | decode spread 0.18%
effect:           decode +0.49%, prefill +0.05% vs base
stats:            decode p=0.0476 q=0.0741; prefill p=0.4286 q=0.5411 | two-sided exact permutation 5 vs 5, BH over the declared family m=28 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  its +1.41% decode at 4x64K (q=0.025; +0.49% at 1x255K) grows past +2% on a later substrate
notes:            Round r1 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Data: data/raw/night-20260919/binrun.tsv, br-single-dpp-warp-reductions-*.md, binstats output in the same folder.
