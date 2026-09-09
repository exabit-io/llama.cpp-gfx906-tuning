# qwen38-27b-ud-q4_k_m  2026-09-03T13:42:41+00:00  model=/root/models/Qwen3.8-27B-UD-Q4_K_M.gguf
# build: ggml_cuda_init: found 4 ROCm devices (Total VRAM: 131008 MiB):
# flags: --device rocm0,rocm0/rocm1,rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 256 -d 0,16384,32768 -r 2
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        201.74 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         24.71 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        | pp2048 @ d16384 |        175.29 ± 0.47 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |  tg256 @ d16384 |         22.52 ± 0.03 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        | pp2048 @ d32768 |        155.49 ± 0.25 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |  tg256 @ d32768 |         20.95 ± 0.03 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |          pp2048 |        383.84 ± 0.27 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |           tg256 |         35.52 ± 0.23 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  | pp2048 @ d16384 |        335.83 ± 0.75 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |  tg256 @ d16384 |         33.59 ± 0.18 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  | pp2048 @ d32768 |        299.58 ± 0.46 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |  tg256 @ d32768 |         32.11 ± 0.16 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        732.92 ± 1.04 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         47.33 ± 1.41 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 | pp2048 @ d16384 |        641.15 ± 3.49 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |  tg256 @ d16384 |         45.60 ± 0.97 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 | pp2048 @ d32768 |        569.79 ± 2.88 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |  tg256 @ d32768 |         44.32 ± 0.80 |

build: 360e134 (1)
# exit=0  2026-09-03T13:57:00+00:00
