patch:            meta-concurrent-lanes | https://github.com/mxxm-t/mx-llama.cpp | (runtime switch GGML_META_PARALLEL_DISPATCH=0, no code change) | issue the meta backend's subgraph lanes concurrently (substrate 5d9efc8ca; removed via GGML_META_PARALLEL_DISPATCH=0, measured on its increment over the token-graph-off arm)
axis:             multi-user
zero point:       build-ps-no-token-graph = master 82868aa3b + no-token-graph measured 2026-09-26/27 (r2, interleaved in the same blocks), kernel 7.0.0-34-generic #34~24.04.1+barfix2
recipe:           4x64K (4 x 65536 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level mean of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode mean 22.556 (median 22.563 / p10 22.529 / min 22.513) vs ref 22.684 tok/s | prefill mean 770.9 (min 770.3) vs ref 771.4 t/s | n=5 per arm | decode spread 0.32%
effect:           feature: decode +0.57%, prefill +0.06% (removal arm measured: switching it off gives -0.57% / -0.06% vs no-token-graph; ref = feature on)
stats:            decode p=0.0079 q=0.0416; prefill p=0.3651 q=0.5805 | two-sided exact permutation 5 vs 5, BH over the declared family m=32 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  its increment on top of the token graph (+0.57% at 4x64K, +0.35% at 1x255K) grows past 2%; it is also the token graph's prerequisite, so it stays on as long as the token graph does
notes:            Round r2 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts (of the FEATURE; the arm switched it off): multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Substrate feature: it stays in master (the substrate stays whole, plan s7); neutral-drop here means no
                  measurable value on this instrument, NOT a removal. Nothing is switched off.
                  Data: data/raw/night-20260919/binrun-r2.tsv, br-r2-multi-no-parallel-dispatch-*.md, binstats output in the same folder.
