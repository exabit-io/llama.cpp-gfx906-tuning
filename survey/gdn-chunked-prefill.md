patch:            gdn-chunked-prefill | https://github.com/mxxm-t/mx-llama.cpp | f62654251 | chunked gated-delta-net prefill kernel (upstream v0.5.0 has only the serial one) (substrate 55b275262 1248e5e37 4405b44cd 0e3249e9d; removed via switch-off patch f62654251 (reverts conflict))
axis:             multi-user
zero point:       build-ps-base = master 82868aa3b measured 2026-09-26/27 (r2, interleaved in the same blocks), kernel 7.0.0-34-generic #34~24.04.1+barfix2
recipe:           4x64K (4 x 65536 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level mean of n=5; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode mean 23.037 (median 23.052 / p10 23.004 / min 22.996) vs ref 23.014 tok/s | prefill mean 762.4 (min 761.7) vs ref 771.4 t/s | n=5 per arm | decode spread 0.28%
effect:           feature: decode -0.10%, prefill +1.18% (removal arm measured: switching it off gives +0.10% / -1.17% vs base; ref = feature on)
stats:            decode p=0.1032 q=0.2751; prefill p=0.0079 q=0.0416 | two-sided exact permutation 5 vs 5, BH over the declared family m=32 | n=5 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  prefill of a GDN-heavier model or shorter prompts: today it is worth +1.18% prefill at 4x64K and +0.64% at 1x255K, decode unaffected
notes:            Round r2 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts (of the FEATURE; the arm switched it off): multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Substrate feature: it stays in master (the substrate stays whole, plan s7); neutral-drop here means no
                  measurable value on this instrument, NOT a removal. Nothing is switched off.
                  Data: data/raw/night-20260919/binrun-r2.tsv, br-r2-multi-no-gdn-chunked-*.md, binstats output in the same folder.
