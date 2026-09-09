# qwen38-27b-nq-argate  2026-09-08T09:39:51+00:00  fusion build tp4; custom AR gate in rows of n_embd=5120 (libggml-hip: /opt/llama.cpp-mxxm-fh-nq/lib/libggml-hip.so.0)
| variant | env | 1 slot | 2 | 4 | 8 | 16 | tg128 (llama-bench) |
|---|---|---:|---:|---:|---:|---:|---:|
| base | BENCH_VARIANT=base | 48.89 | 77.05 | 121.98 | 175.00 | 203.86 |  - |
| pw-default | GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 | 55.82 | 92.18 | 124.85 | 160.11 | 187.39 |  - |
| pw-1row | GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=5121 | 52.12 | 76.35 | 121.14 | 175.33 | 203.93 |  - |
| pw-2rows | GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=10241 | 44.85 | 92.88 | 122.12 | 175.72 | 203.92 |  - |
| pw-4rows | GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481 | 54.26 | 91.70 | 123.64 | 174.91 | 204.11 |  - |
| pw-8rows | GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=40961 | 54.59 | 92.98 | 125.50 | 166.56 | 203.81 |  - |
| base-repeat | BENCH_VARIANT=base2 | 48.82 | 76.97 | 122.46 | 174.83 | 203.72 |  - |
# done 2026-09-08T09:52:57+00:00
