| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_0                |  14.94 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        329.34 ± 4.58 |
| qwen35 27B Q4_0                |  14.94 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         27.13 ± 0.02 |
| qwen35 27B Q4_0                |  14.94 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |       1111.65 ± 2.08 |
| qwen35 27B Q4_0                |  14.94 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         50.82 ± 1.65 |

build: 360e134 (1)
