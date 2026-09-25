patch:            max-ilp | https://github.com/mixa3607/ML-gfx906 | (build flag) | compile with -mllvm -amdgpu-sched-strategy=max-ilp (mixa3607/ML-gfx906), base code unchanged (build flag)
axis:             multi-user
zero point:       build-ps-base = master a23e12438 measured 2026-09-24/25 (r1, interleaved in the same blocks)
recipe:           4x64K (4 x 65536 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level mean of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode mean 22.699 (median 22.697 / p10 22.681 / min 22.677) vs ref 22.468 tok/s | prefill mean 772.0 (min 771.2) vs ref 760.9 t/s | n=5 per arm | decode spread 0.23%
effect:           decode +1.03%, prefill +1.47% vs base
stats:            decode p=0.0079 q=0.0247; prefill p=0.0079 q=0.0247 | two-sided exact permutation 5 vs 5, BH over the declared family m=28 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  a later ROCm/LLVM changes the max-ilp scheduler: today +1.03% / +1.47% (decode/prefill) at 4x64K and +0.74% / +0.84% at 1x255K, all q<0.10, none reaching 2%
notes:            Round r1 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts: multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Data: data/raw/night-20260919/binrun.tsv, br-multi-max-ilp-*.md, binstats output in the same folder.
