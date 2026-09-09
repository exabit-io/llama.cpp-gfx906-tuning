# qwen38-27b-q8_0-opt C: llama-bench -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 256 -d 0 -r 2, one die and four  2026-09-06T19:08:27+00:00
## 27b-q4_0 (/root/models/Qwen3.8-27B-Q4_0.gguf)
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_0                |  14.94 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        329.34 ± 4.58 |
| qwen35 27B Q4_0                |  14.94 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         27.13 ± 0.02 |
| qwen35 27B Q4_0                |  14.94 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |       1111.65 ± 2.08 |
| qwen35 27B Q4_0                |  14.94 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         50.82 ± 1.65 |
## 27b-q6_k (/root/models/Qwen3.8-27B-UD-Q6_K.gguf)
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        176.89 ± 0.08 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         20.99 ± 0.01 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        649.86 ± 1.13 |
| qwen35 27B Q6_K                |  20.46 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         45.55 ± 1.32 |
## 27b-q4_k_m (/root/models/Qwen3.8-27B-UD-Q4_K_M.gguf)
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        734.88 ± 1.18 |
| qwen35 27B Q4_K - Medium       |  15.32 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         47.46 ± 1.45 |
## 9b-q8_0 (/root/models/Qwen3.5-9B-Q8_0.gguf)
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |       768.37 ± 13.90 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         56.52 ± 0.02 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |          pp2048 |     1300.28 ± 157.80 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |           tg256 |         76.32 ± 0.85 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |     1911.57 ± 873.71 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |        100.88 ± 3.20 |
## 2b-q8_0 (/root/models/Qwen3.5-2B-Q8_0.gguf)
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 2B Q8_0                 |   1.86 GiB |     1.88 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |     2491.43 ± 968.20 |
| qwen35 2B Q8_0                 |   1.86 GiB |     1.88 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |        161.98 ± 0.16 |
| qwen35 2B Q8_0                 |   1.86 GiB |     1.88 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |          pp2048 |    4076.52 ± 2299.86 |
| qwen35 2B Q8_0                 |   1.86 GiB |     1.88 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |           tg256 |        176.04 ± 3.03 |
| qwen35 2B Q8_0                 |   1.86 GiB |     1.88 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |    4979.15 ± 4519.47 |
| qwen35 2B Q8_0                 |   1.86 GiB |     1.88 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |        171.41 ± 6.71 |
## 0.8b-q8_0 (/root/models/Qwen3.5-0.8B-Q8_0.gguf)
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |    4093.58 ± 2394.85 |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |        171.36 ± 0.49 |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |          pp2048 |     10527.24 ± 68.88 |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1  |           tg256 |        214.54 ± 6.07 |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |    13539.79 ± 275.35 |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |        194.75 ± 9.32 |
# done 2026-09-06T19:17:27+00:00
