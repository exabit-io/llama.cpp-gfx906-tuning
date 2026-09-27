patch:            meta-concurrent-lanes | https://github.com/mxxm-t/mx-llama.cpp | (runtime switch GGML_META_PARALLEL_DISPATCH=0, no code change) | issue the meta backend's subgraph lanes concurrently (substrate 5d9efc8ca; removed via GGML_META_PARALLEL_DISPATCH=0, measured on its increment over the token-graph-off arm)
axis:             single-user
zero point:       build-ps-no-token-graph = master 82868aa3b + no-token-graph measured 2026-09-26/27 (r2, interleaved in the same blocks), kernel 7.0.0-34-generic #34~24.04.1+barfix2
recipe:           1x255K (1 x 260864 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level median of n=6; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode median 30.852 (median 30.852 / p10 30.806 / min 30.784) vs ref 30.959 tok/s | prefill median 401.2 (min 401.0) vs ref 401.3 t/s | n=6 per arm | decode spread 0.38%
effect:           feature: decode +0.35%, prefill +0.03% (removal arm measured: switching it off gives -0.35% / -0.03% vs no-token-graph; ref = feature on)
stats:            decode p=0.0130 q=0.0416; prefill p=0.6494 q=0.8533 | two-sided exact permutation 6 vs 6, BH over the declared family m=32 | n=6 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  its increment on top of the token graph (+0.57% at 4x64K, +0.35% at 1x255K) grows past 2%; it is also the token graph's prerequisite, so it stays on as long as the token graph does
notes:            Round r2 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts (of the FEATURE; the arm switched it off): multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Substrate feature: it stays in master (the substrate stays whole, plan s7); neutral-drop here means no
                  measurable value on this instrument, NOT a removal. Nothing is switched off.
                  Data: data/raw/night-20260919/binrun-r2.tsv, br-r2-single-no-parallel-dispatch-*.md, binstats output in the same folder.
