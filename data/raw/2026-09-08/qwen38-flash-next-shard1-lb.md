| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |         lm |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | ---------: | --------------: | -------------------: |
| qwen4exp A3B Q4_K - Medium     | 103.68 GiB |   176.94 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |        dio |          pp2048 |        365.24 ± 0.52 |
| qwen4exp A3B Q4_K - Medium     | 103.68 GiB |   176.94 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |        dio |           tg128 |         20.64 ± 8.00 |

build: 5b7794476 (11029)
