patch:            rccl-collective | GGML_HIP_RCCL=ON build option | R3.11 | use RCCL for tensor-parallel collectives instead of the meta-backend butterfly fallback
axis:             multi-user
zero point:       build-c4series-rccl with GGML_CUDA_ALLREDUCE=none (butterfly) measured 2026-09-21/22
recipe:           4x64K, 1x254K, 1x64K | q8_0 K and V | 125 W/die | --cache-ram 49152 | -ngl all | one binary, GGML_CUDA_ALLREDUCE selects the path
metric:           prefill t/s, derived at full precision (tools/cell-metrics.py); improves requires q<0.10 and effect >= +2%
result:           4x64K 633.1 -> 750.8 | 1x254K 355.0 -> 395.2 | 1x64K 631.6 -> 746.6 t/s (n=4 per arm per cell)
effect:           prefill +18.59% multi-user, +11.34% single-user 254K, +18.21% single-user 64K; decode unaffected
stats:            all three p=0.0286-0.0571, q=0.0857 PASS | BH over m=9 | n=4 fresh per arm per cell
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
