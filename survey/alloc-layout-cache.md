patch:            alloc-layout-cache | https://github.com/mxxm-t/mx-llama.cpp | (runtime switch GGML_GALLOC_LAYOUT_CACHE=0, no code change) | cache and restore graph-allocator layouts (substrate 6d2012d8a 42b3cfb63 687ef0194 d5047d6aa; removed via GGML_GALLOC_LAYOUT_CACHE=0)
axis:             multi-user
zero point:       build-ps-base = master 82868aa3b measured 2026-09-26/27 (r2, interleaved in the same blocks), kernel 7.0.0-34-generic #34~24.04.1+barfix2
recipe:           4x64K (4 x 65536 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level mean of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode mean 23.019 (median 23.017 / p10 22.986 / min 22.970) vs ref 23.014 tok/s | prefill mean 771.6 (min 770.9) vs ref 771.4 t/s | n=5 per arm | decode spread 0.40%
effect:           feature: decode -0.02%, prefill -0.03% (removal arm measured: switching it off gives +0.02% / +0.03% vs base; ref = feature on)
stats:            decode p=0.7857 q=0.9670; prefill p=0.4127 q=0.6003 | two-sided exact permutation 5 vs 5, BH over the declared family m=32 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  graph re-allocation becomes a measurable share of a token (e.g. graphs that change shape every step); today switching it off changes nothing (+0.02% / +0.07%)
notes:            Round r2 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts (of the FEATURE; the arm switched it off): multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Substrate feature: it stays in master (the substrate stays whole, plan s7); neutral-drop here means no
                  measurable value on this instrument, NOT a removal. Nothing is switched off.
                  Data: data/raw/night-20260919/binrun-r2.tsv, br-r2-multi-no-layout-cache-*.md, binstats output in the same folder.
