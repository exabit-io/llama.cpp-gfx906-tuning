| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |    4093.58 ± 2394.85 |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |        171.36 ± 0.49 |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |          pp2048 |     10527.24 ± 68.88 |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |           tg256 |        214.54 ± 6.07 |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |    13539.79 ± 275.35 |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |        194.75 ± 9.32 |

build: 360e134 (1)
