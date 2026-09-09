| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        179.00 ± 0.02 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg128 |         19.96 ± 1.58 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        670.88 ± 0.20 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg128 |         46.92 ± 2.26 |

build: c36c28ed2 (10291)
