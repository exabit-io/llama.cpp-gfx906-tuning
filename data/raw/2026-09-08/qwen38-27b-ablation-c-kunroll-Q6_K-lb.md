| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        179.10 ± 0.07 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg128 |         19.96 ± 1.62 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        671.06 ± 0.34 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg128 |         47.02 ± 2.27 |

build: 6c2c03a95 (10292)
