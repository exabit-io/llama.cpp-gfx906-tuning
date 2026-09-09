| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 2B Q8_0                 |   1.86 GiB |     1.88 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |     2491.43 ± 968.20 |
| qwen35 2B Q8_0                 |   1.86 GiB |     1.88 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |        161.98 ± 0.16 |
| qwen35 2B Q8_0                 |   1.86 GiB |     1.88 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |          pp2048 |    4076.52 ± 2299.86 |
| qwen35 2B Q8_0                 |   1.86 GiB |     1.88 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |           tg256 |        176.04 ± 3.03 |
| qwen35 2B Q8_0                 |   1.86 GiB |     1.88 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |    4979.15 ± 4519.47 |
| qwen35 2B Q8_0                 |   1.86 GiB |     1.88 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |        171.41 ± 6.71 |

build: 360e134 (1)
