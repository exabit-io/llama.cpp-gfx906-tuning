| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |       768.37 ± 13.90 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         56.52 ± 0.02 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |          pp2048 |     1300.28 ± 157.80 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |           tg256 |         76.32 ± 0.85 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |     1911.57 ± 873.71 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |        100.88 ± 3.20 |

build: 360e134 (1)
