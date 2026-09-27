patch:            q8_1-activation-cache | https://github.com/mxxm-t/mx-llama.cpp | (runtime switch GGML_CUDA_Q8_1_CACHE=0, no code change) | reuse the quantized Q8_1 activation across matmuls in one graph (incl. the q8_repack path) (substrate 775a8051f cfea4a1f6 0a694d8ad; removed via GGML_CUDA_Q8_1_CACHE=0)
axis:             single-user
zero point:       build-ps-base = master 82868aa3b measured 2026-09-26/27 (r2, interleaved in the same blocks), kernel 7.0.0-34-generic #34~24.04.1+barfix2
recipe:           1x255K (1 x 260864 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level median of n=6; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode median 31.496 (median 31.496 / p10 31.444 / min 31.435) vs ref 31.639 tok/s | prefill median 401.6 (min 401.3) vs ref 401.5 t/s | n=6 per arm | decode spread 0.36%
effect:           feature: decode +0.45%, prefill -0.03% (removal arm measured: switching it off gives -0.45% / +0.03% vs base; ref = feature on)
stats:            decode p=0.0130 q=0.0416; prefill p=0.6667 q=0.8533 | two-sided exact permutation 6 vs 6, BH over the declared family m=32 | n=6 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  the cache's decode value grows past 2%: today +1.61% at 4x64K (q=0.042) and +0.45% at 1x255K — it pays more with more rows in flight, so a wider multi-user cell would be the place to look
notes:            Round r2 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts (of the FEATURE; the arm switched it off): multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Substrate feature: it stays in master (the substrate stays whole, plan s7); neutral-drop here means no
                  measurable value on this instrument, NOT a removal. Nothing is switched off.
                  Data: data/raw/night-20260919/binrun-r2.tsv, br-r2-single-no-q8_1-cache-*.md, binstats output in the same folder.
