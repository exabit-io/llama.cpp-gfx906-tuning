patch:            gfx906-mmq-kernel-tweaks | https://github.com/mxxm-t/mx-llama.cpp | 2032a34e6 | gfx906 Q8_0 MMQ tile-load threads-per-row 8 and vec_dot unroll-2 (substrate 5c4505b5d f1684f76c; removed via switch-off patch 2032a34e6)
axis:             multi-user
zero point:       build-ps-base = master 82868aa3b measured 2026-09-26/27 (r2, interleaved in the same blocks), kernel 7.0.0-34-generic #34~24.04.1+barfix2
recipe:           4x64K (4 x 65536 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level mean of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode mean 23.028 (median 23.040 / p10 22.997 / min 22.991) vs ref 23.014 tok/s | prefill mean 771.8 (min 771.1) vs ref 771.4 t/s | n=5 per arm | decode spread 0.31%
effect:           feature: decode -0.06%, prefill -0.05% (removal arm measured: switching it off gives +0.06% / +0.05% vs base; ref = feature on)
stats:            decode p=0.3730 q=0.5805; prefill p=0.2381 q=0.5419 | two-sided exact permutation 5 vs 5, BH over the declared family m=32 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  a workload whose Q8_0 matmuls run through MMQ instead of q8_repack (repack off, or a non-repacked type); here they are inert (+-0.1%)
notes:            Round r2 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts (of the FEATURE; the arm switched it off): multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Substrate feature: it stays in master (the substrate stays whole, plan s7); neutral-drop here means no
                  measurable value on this instrument, NOT a removal. Nothing is switched off.
                  Data: data/raw/night-20260919/binrun-r2.tsv, br-r2-multi-no-mmq-kernel-tweaks-*.md, binstats output in the same folder.
