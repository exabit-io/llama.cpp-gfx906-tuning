patch:            q8_1-activation-cache | https://github.com/mxxm-t/mx-llama.cpp | (runtime switch GGML_CUDA_Q8_1_CACHE=0, no code change) | reuse the quantized Q8_1 activation across matmuls in one graph (incl. the q8_repack path) (substrate 775a8051f cfea4a1f6 0a694d8ad; removed via GGML_CUDA_Q8_1_CACHE=0)
axis:             multi-user
zero point:       build-ps-base = master 82868aa3b measured 2026-09-26/27 (r2, interleaved in the same blocks), kernel 7.0.0-34-generic #34~24.04.1+barfix2
recipe:           4x64K (4 x 65536 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level mean of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode mean 22.649 (median 22.653 / p10 22.625 / min 22.611) vs ref 23.014 tok/s | prefill mean 771.3 (min 771.0) vs ref 771.4 t/s | n=5 per arm | decode spread 0.29%
effect:           feature: decode +1.61%, prefill +0.01% (removal arm measured: switching it off gives -1.59% / -0.01% vs base; ref = feature on)
stats:            decode p=0.0079 q=0.0416; prefill p=0.8810 q=0.9896 | two-sided exact permutation 5 vs 5, BH over the declared family m=32 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  the cache's decode value grows past 2%: today +1.61% at 4x64K (q=0.042) and +0.45% at 1x255K — it pays more with more rows in flight, so a wider multi-user cell would be the place to look
notes:            Round r2 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts (of the FEATURE; the arm switched it off): multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Substrate feature: it stays in master (the substrate stays whole, plan s7); neutral-drop here means no
                  measurable value on this instrument, NOT a removal. Nothing is switched off.
                  Data: data/raw/night-20260919/binrun-r2.tsv, br-r2-multi-no-q8_1-cache-*.md, binstats output in the same folder.
