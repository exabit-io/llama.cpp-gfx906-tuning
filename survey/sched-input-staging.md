patch:            sched-input-staging | https://github.com/mxxm-t/mx-llama.cpp | -20af9a480 -c7069d868 -38e266b6a | sched: geometric input staging buffers, input ring sized to the pipeline width, view inputs staged as views (substrate 38e266b6a c7069d868 20af9a480; removed via git revert)
axis:             multi-user
zero point:       build-ps-base = master 82868aa3b measured 2026-09-26/27 (r2, interleaved in the same blocks), kernel 7.0.0-34-generic #34~24.04.1+barfix2
recipe:           4x64K (4 x 65536 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level mean of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode mean 23.025 (median 23.027 / p10 22.995 / min 22.980) vs ref 23.014 tok/s | prefill mean 769.5 (min 768.4) vs ref 771.4 t/s | n=5 per arm | decode spread 0.36%
effect:           feature: decode -0.05%, prefill +0.24% (removal arm measured: switching it off gives +0.05% / -0.24% vs base; ref = feature on)
stats:            decode p=0.4921 q=0.6846; prefill p=0.0079 q=0.0416 | two-sided exact permutation 5 vs 5, BH over the declared family m=32 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  pipeline-parallel (-sm layer) or multi-stage runs, which its claims (+37-69% layer prefill) target; on this single-stage instrument it is worth +0.24% / +0.90% prefill
notes:            Round r2 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts (of the FEATURE; the arm switched it off): multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text different vs base).
                  Substrate feature: it stays in master (the substrate stays whole, plan s7); neutral-drop here means no
                  measurable value on this instrument, NOT a removal. Nothing is switched off.
                  The text was read: identical to base up to the last clause ("shorter wavelengths like blue and violet are"
                  vs "(blue and violet)"), coherent — a greedy-decode near-tie (the same flip gdn-producer-fold showed in round 1).
                  Data: data/raw/night-20260919/binrun-r2.tsv, br-r2-multi-revert-input-staging-*.md, binstats output in the same folder.
