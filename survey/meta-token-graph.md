patch:            meta-token-graph | https://github.com/mxxm-t/mx-llama.cpp | (runtime switch GGML_META_TOKEN_GRAPH=0, no code change) | record a whole decode token into one graph per lane and replay it (substrate 751b6114c a3ab67c89 693375a1f a355590d2; removed via GGML_META_TOKEN_GRAPH=0)
axis:             multi-user
zero point:       build-ps-base = master 82868aa3b measured 2026-09-26/27 (r2, interleaved in the same blocks), kernel 7.0.0-34-generic #34~24.04.1+barfix2
recipe:           4x64K (4 x 65536 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level mean of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode mean 22.684 (median 22.684 / p10 22.680 / min 22.678) vs ref 23.014 tok/s | prefill mean 771.4 (min 770.5) vs ref 771.4 t/s | n=5 per arm | decode spread 0.05%
effect:           feature: decode +1.45%, prefill -0.01% (removal arm measured: switching it off gives -1.43% / +0.01% vs base; ref = feature on)
stats:            decode p=0.0079 q=0.0416; prefill p=0.8968 q=0.9896 | two-sided exact permutation 5 vs 5, BH over the declared family m=32 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              single-user-only
would change if:  single-user decode stops depending on per-kernel launch overhead (e.g. a faster HIP launch path): today the token graph is worth +2.20% at 1x255K (31.64 vs 30.96 without it, q=0.042) and +1.45% at 4x64K; together with concurrent lanes +2.0% / +2.6%
notes:            Round r2 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts (of the FEATURE; the arm switched it off): multi-user neutral, single-user improves. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Already on by default in master (GGML_META_TOKEN_GRAPH unset = on): nothing to change. It improves single-user
                  only; on multi-user it is neutral (+1.45%), so it is not switched off there either.
                  Data: data/raw/night-20260919/binrun-r2.tsv, br-r2-multi-no-token-graph-*.md, binstats output in the same folder.
