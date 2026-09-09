| model                          |       size |     params | backend    | ngl |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg128 |         49.40 ± 1.34 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 | tensor |   1 | ROCm0        |           tg128 |         20.45 ± 0.04 |

build: a1462e0 (9)
