patch:            rccl-collective | GGML_HIP_RCCL=ON build option | R3.11 | use RCCL for tensor-parallel collectives instead of the meta-backend butterfly fallback
axis:             single-user
zero point:       build-c4series-rccl with GGML_CUDA_ALLREDUCE=none (butterfly) measured 2026-09-21/22
recipe:           1 x 254K PRIMARY, 1 x 64K control | -ctk q8_0 | 125 W/die | --cache-ram 49152 | -ngl all | RCCL + gated custom AR
metric:           single-stream decode tok/s (prefill secondary), MEDIAN of n>=4 — the four-die single-stream stall fires about 1 run in 16 and shifts a mean by ~1.5%, most of the 2% floor; improves requires q<0.10 and effect >= +2%
result:           prefill median 355.0 -> 395.2 (254K) and 631.6 -> 746.6 (64K) t/s, n=4 per arm
effect:           prefill +11.34% at 254K, +18.21% at 64K; decode unaffected
stats:            p=0.0571 q=0.0857 PASS on both cells | BH over m=9 | n=4 fresh per arm
evidence:         confirmed-fresh
structural:       standalone
verdict:          improves
bin:              both
would change if:  a deeper cell shows RCCL's channel setup cost dominating, or RCCL regresses at slot counts above 4
notes:            This is the largest confirmed effect in the campaign and it is a BUILD FLAG, not a patch.
                  GGML_HIP_RCCL defaults OFF upstream and was inherited unaudited, so every earlier
                  measurement ran on the butterfly fallback (init chain: nccl -> internal, which needs
                  n_devices==2 -> none -> butterfly). R3.11 now requires it.
                  Custom AR and the collective are ORTHOGONAL: custom AR owns decode (+5-9%) and has no
                  measurable effect on prefill (+/-0.06%, n.s. on all three cells); the collective owns
                  prefill and does not touch decode. They ship together.

AXIS NOTE:        companion record to rccl-collective.md. R2.7 (corrected 2026-09-21) requires the single-user
                  axis to be measured at its OWN design point: 1 x 254K primary, 1 x 64K control.
                  Any single-user figure in this campaign dated before that correction was taken at
                  1 x 32K and is superseded -- a floor is not a design point.
