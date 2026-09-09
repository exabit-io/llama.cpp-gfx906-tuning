| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |   nr |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | ---: | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |    1 |          pp2048 |       1137.04 ± 0.93 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |    1 |           tg128 |        37.91 ± 16.39 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |    1 |          pp2048 |        338.75 ± 0.02 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |    1 |           tg128 |         21.43 ± 0.10 |

build: 5d36d6fc8 (11032)
