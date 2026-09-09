| model                          |       size |     params | backend    | ngl |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg128 |         50.27 ± 1.36 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 | tensor |   1 | ROCm0        |           tg128 |         21.60 ± 0.05 |

build: e6938f2 (6)
