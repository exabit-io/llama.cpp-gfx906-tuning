# qwen35-9b-q8_0  2026-09-03T09:20:37+00:00  model=/root/models/Qwen3.5-9B-Q8_0.gguf
# build: ggml_cuda_init: found 4 ROCm devices (Total VRAM: 131008 MiB):
# flags: --device rocm0,rocm0/rocm1,rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 256 -d 0,16384,32768 -r 3
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        776.95 ± 0.87 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         56.76 ± 0.04 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        | pp2048 @ d16384 |        660.10 ± 1.36 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |  tg256 @ d16384 |         53.94 ± 0.07 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        | pp2048 @ d32768 |        570.16 ± 1.07 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |  tg256 @ d32768 |         51.68 ± 0.07 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |          pp2048 |       1389.42 ± 1.98 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |           tg256 |         75.71 ± 0.47 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  | pp2048 @ d16384 |       1179.03 ± 1.92 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |  tg256 @ d16384 |         72.78 ± 0.39 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  | pp2048 @ d32768 |       1058.77 ± 3.26 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |  tg256 @ d32768 |         70.51 ± 0.34 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |      2494.72 ± 33.99 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |        101.00 ± 3.17 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 | pp2048 @ d16384 |      2154.09 ± 16.31 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |  tg256 @ d16384 |         97.69 ± 1.53 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 | pp2048 @ d32768 |      1890.57 ± 13.60 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |  tg256 @ d32768 |         95.56 ± 1.44 |

build: 360e134 (1)
# exit=0  2026-09-03T09:26:06+00:00
