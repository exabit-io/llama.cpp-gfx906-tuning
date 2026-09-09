# qwen38-27b-mxxm  2026-09-07T17:34:09+00:00  mxxm (mx-llama.cpp b10254 + kcase patch) vs stock b10288; llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4

## mxxm: /opt/llama.cpp-mxxm/bin/llama-bench
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        314.97 ± 9.73 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         20.78 ± 0.03 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |       1129.62 ± 1.47 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         49.06 ± 0.84 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        228.71 ± 0.10 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         21.01 ± 0.03 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        827.39 ± 0.09 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         49.35 ± 0.88 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        216.65 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         24.99 ± 0.04 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        796.11 ± 0.73 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         51.83 ± 0.92 |

## stock: /opt/llama.cpp/bin/llama-bench
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        233.62 ± 0.86 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         20.59 ± 0.01 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        848.44 ± 1.42 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         47.28 ± 1.42 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        174.98 ± 0.05 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         20.99 ± 0.01 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        658.05 ± 0.70 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         47.47 ± 1.44 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        201.26 ± 0.15 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         24.86 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        745.95 ± 1.26 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         49.71 ± 1.56 |
# done 2026-09-07T17:46:28+00:00
