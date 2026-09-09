| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        176.69 ± 0.07 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg128 |         20.91 ± 0.02 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        661.78 ± 0.18 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg128 |         46.94 ± 2.25 |

build: 39c917808 (10289)
