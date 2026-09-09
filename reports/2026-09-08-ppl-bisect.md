| build | commit | ppl 16K/6 |
|---|---|---|
| master 5d806aa25 pristine (/opt/llama.cpp-master) | 2026-09-08 | 5.6216 ± 0.06246 |
| gfx906 branch (fork b10912 + merge + series, /opt/llama.cpp-gfx906-master) | 2026-09-08 | 5.6173 ± 0.06235 |
| production (fork b10254 + series) | | 5.5969 |
| 0ba6499c3 | 2026-09-03 CUDA: Allow concurrent streams per split for multi-GPU (#28198) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| 3ad1ba733 | 2026-09-06 [Model] Support for Spark2_5ForCausalLM  implementation (#27868) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| 5a6caa05f | 2026-09-08 ggml : update ggml_prec specification (#26675) | Final estimate: PPL = 5.6216 +/- 0.06246 |
| 5fdfa6282 | 2026-09-06 models : fix GDN normalization from `max` to `rsqrt` (#28068) | Final estimate: PPL = 5.6216 +/- 0.06246 |
| 64a155d24 | 2026-09-04 sync : ggml (#28379) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| 73ab7599b | 2026-09-07 CUDA: branchless Q4_K/Q5_K unpack to speed up mmvq, L2 prefetch on DGX Spark (# | Final estimate: PPL = 5.6216 +/- 0.06246 |
| 8e93a9773 | 2026-09-02 CUDA + ggml: add sparse-fa for DSV4/GLM (#27970) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| 992cb503c | 2026-09-07 ggml: allow backend inputs to not create another split (#28387) | Final estimate: PPL = 5.6216 +/- 0.06246 |
| 0f3a71be1 | 2026-09-02 mtmd: Fix Qwen3-tts-0.6b (#28231) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| llama.cpp-b10912 |  | Final estimate: PPL = 5.6143 +/- 0.06230 |
| b10837-5202104 |  | Final estimate: PPL = 5.6216 +/- 0.06246 |

## step-1 bisect (upstream 360e1349f 2026-08-05 = stock b10288, 5.5969 .. 0f3a71be1 2026-09-02, 5.6118)
| 4c6766fd7 | 2026-08-10 vendor: sync subprocess.h and drop local patches (#26808) | Final estimate: PPL = 5.5969 +/- 0.06197 |
| decaf508b | 2026-08-13 server: refactor + correctness fixes for metrics (#26920) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| d83f72d46 | 2026-08-17 ci : restore release.yml check during make-release.yml (#27247) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| 9ee9fc04c | 2026-08-19 opencl: make the MoE expert scatter deterministic (#26464) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| 9fee29e94 | 2026-08-22 arg: remove -no-cnv from cli [no ci] (#27542) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| 925e11799 | 2026-08-26 llama: add token ID tracking to KV cell (#27762) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| cc231cb0d | 2026-08-30 dflash: pass missing NVFP4 scales to attention operations (#28000) | Final estimate: PPL = 5.6118 +/- 0.06229 |

## step-1 pin: around e79e4bf66 (remove -funsafe-math-optimizations from the HIP build)
| d86c7d62d | 2026-08-13 ui: Clean up contexts, remove prop drilling from Chat Form Actions (#26951) | Final estimate: PPL = 5.5969 +/- 0.06197 |
| e79e4bf66 | 2026-08-12 ggml-hip : remove -funsafe-math-optimizations (#26696) | Final estimate: PPL = 5.6118 +/- 0.06229 |
| r3-unsafe-math |  | Final estimate: PPL = 5.6212 +/- 0.06248 |
