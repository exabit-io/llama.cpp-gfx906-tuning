| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        199.88 ± 4.58 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg128 |         24.83 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        750.86 ± 0.70 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg128 |         49.05 ± 2.51 |

build: c36c28ed2 (10291)
