# qwen38-27b-mxxmfh-forkknobs  2026-09-08T09:04:45+00:00  fork knobs on the production build, tp4 (libggml-hip: /opt/llama.cpp-mxxm-fh/lib/libggml-hip.so.0)
| variant | env | AR path (from the log) | pp2048 t/s | tg128 t/s | 8 slots 2K t/s | 16 slots 2K t/s |
|---|---|---|---:|---:|---:|---:|
| baseline | BENCH_VARIANT=baseline | RCCL (no custom AR) | pp2048 | tg128 | 174.49 | 203.76 |
| custom-ar-oneshot | GGML_ENABLE_CUSTOM_AR=1 | RCCL (no custom AR) | pp2048 | tg128 | 174.25 | 204.14 |
| peerwrite | GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 | RCCL (no custom AR) | pp2048 | tg128 | 160.16 | 187.84 |
| peerwrite-no-tg | GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_META_TOKEN_GRAPH=0 | RCCL (no custom AR) | pp2048 | tg128 | 155.06 | 182.06 |
| peerwrite-nogate | GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_NO_GATE=1 | RCCL (no custom AR) | pp2048 | tg128 | 158.47 | 185.03 |
| fgp-only | HSA_FORCE_FINE_GRAIN_PCIE=1 | RCCL (no custom AR) | pp2048 | tg128 | 174.82 | 204.30 |
| serial-dispatch | GGML_META_PARALLEL_DISPATCH=0 | RCCL (no custom AR) | pp2048 | tg128 | 173.86 | 203.12 |
| baseline-repeat | BENCH_VARIANT=baseline2 | RCCL (no custom AR) | pp2048 | tg128 | 174.55 | 204.07 |

Perplexity, peer-write custom AR, tp4 -c 16384 --chunks 6: Final estimate: PPL = 5.5969 +/- 0.06197 (reference 5.5969 +/- 0.062 on the same build with RCCL)

```
== Agent 1: 65 tokens (delimiter void mul_mat_vec_q<(ggml_type)8, 1, false, false>(void const... grid 3973120), steady window 55 tokens, median per token:
  token period (delimiter to delimiter) 22.96 ms; inter-token gap 0.49 ms
  kernels 1866   span 22.47 ms   busy(union) 21.68 ms   idle-in-span 0.75 ms
  mmvq: 11.04 ms/433  mmq: 0.00 ms/0  fa: 0.37 ms/32  rccl: 4.97 ms/128  norm: 1.46 ms/305  cpy: 0.28 ms/70  other: 3.56 ms/898  rt: 0.00 ms/0
  gaps per token: 0-10us: 0 (0.00 ms), 10-30us: 2 (0.03 ms), 30-100us: 1 (0.06 ms), 100-300us: 2 (0.30 ms), 300-1000000000us: 1 (0.34 ms)
  token period (delimiter to delimiter) 22.97 ms; inter-token gap 0.52 ms
  kernels 1866   span 22.45 ms   busy(union) 20.44 ms   idle-in-span 1.95 ms
  mmvq: 11.14 ms/433  mmq: 0.00 ms/0  fa: 0.37 ms/32  rccl: 3.65 ms/128  norm: 1.44 ms/305  cpy: 0.28 ms/70  other: 3.56 ms/898  rt: 0.00 ms/0
  gaps per token: 0-10us: 0 (0.00 ms), 10-30us: 1 (0.02 ms), 30-100us: 2 (0.12 ms), 100-300us: 2 (0.20 ms), 300-1000000000us: 1 (1.70 ms)
  token period (delimiter to delimiter) 22.97 ms; inter-token gap 0.56 ms
  kernels 1866   span 22.40 ms   busy(union) 19.34 ms   idle-in-span 2.99 ms
  mmvq: 10.99 ms/433  mmq: 0.00 ms/0  fa: 0.37 ms/32  rccl: 2.70 ms/128  norm: 1.44 ms/305  cpy: 0.28 ms/70  other: 3.56 ms/898  rt: 0.00 ms/0
```
# done 2026-09-08T09:22:27+00:00
