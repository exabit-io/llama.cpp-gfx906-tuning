patch:            gdn-chunked-prefill | https://github.com/mxxm-t/mx-llama.cpp | f62654251 | chunked gated-delta-net prefill kernel (upstream v0.5.0 has only the serial one) (substrate 55b275262 1248e5e37 4405b44cd 0e3249e9d; removed via switch-off patch f62654251 (reverts conflict))
axis:             single-user
zero point:       build-ps-base = master 82868aa3b measured 2026-09-26/27 (r2, interleaved in the same blocks), kernel 7.0.0-34-generic #34~24.04.1+barfix2
recipe:           1x255K (1 x 260864 prompt, 1024 generated) | f16 K / f16 V | 125 W/die, host RAPL 150 W | --cache-ram n/a (llama-batched-bench) | -ngl all -sm tensor -fa on, four dies | RCCL + custom AR gated 20481, NCCL_TOPO_FILE
metric:           decode tok/s per sequence and prefill t/s, full precision (tools/cell-metrics.py); run-level median of n=6; improves requires q<0.10 AND effect >= +2%, regresses q<0.10 AND <= -2%
result:           decode median 31.681 (median 31.681 / p10 31.598 / min 31.553) vs ref 31.639 tok/s | prefill median 399.0 (min 398.7) vs ref 401.5 t/s | n=6 per arm | decode spread 0.48%
effect:           feature: decode -0.13%, prefill +0.64% (removal arm measured: switching it off gives +0.13% / -0.64% vs base; ref = feature on)
stats:            decode p=0.2814 q=0.5419; prefill p=0.0130 q=0.0416 | two-sided exact permutation 6 vs 6, BH over the declared family m=32 | n=6 per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          neutral
bin:              neutral-drop
would change if:  prefill of a GDN-heavier model or shorter prompts: today it is worth +1.18% prefill at 4x64K and +0.64% at 1x255K, decode unaffected
notes:            Round r2 of the binning plan (RE-RE-SURVEY-ACTION-PLAN s6), binstats.py rule fixed before data.
                  Per-axis verdicts (of the FEATURE; the arm switched it off): multi-user neutral, single-user neutral. Flash-Next compatibility: PASS (rc=0, text identical vs base).
                  Substrate feature: it stays in master (the substrate stays whole, plan s7); neutral-drop here means no
                  measurable value on this instrument, NOT a removal. Nothing is switched off.
                  Data: data/raw/night-20260919/binrun-r2.tsv, br-r2-single-no-gdn-chunked-*.md, binstats output in the same folder.
