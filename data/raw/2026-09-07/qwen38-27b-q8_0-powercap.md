# qwen38-27b-q8_0-powercap  2026-09-07T15:15:22+00:00  GPU power-cap sweep, Qwen3.8-27B Q8_0, stock b10288 build
Per point: llama-bench rocm0 + tp4 (-p 2048 -n 256 -d 0 -r 2), then llama-server tp4 -np 8 -c 262144 with server-bench conc 1,8 (1300-token prompts, 256 gen).
Order: 200, 185, 175, 160, 150, 200 (drift control). Power per phase: slice qwen38-27b-q8_0-powercap-clocks.txt (W per die, 5 s) and smc-power-*.log by the PHASE timestamps in qwen38-27b-q8_0-powercap.progress.

## cap200-1: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        232.87 ± 1.82 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         20.32 ± 0.02 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        847.30 ± 1.14 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         47.41 ± 1.45 |

## cap200-1: llama-server tp4 -np 8 -c 262144, server-bench conc 1,8
# qwen38-27b-q8_0-powercap-cap200-1-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T15:17:43
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 61.7 | 33.2 | 202 | 7.8 | 46.0 | 1.81 | 7.7 |
| 8 | 16 | 62.4 | 65.7 | 399 | 15.4 | 12.7 | 6.41 | 31.1 |

## cap185-2: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        225.93 ± 4.53 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         19.95 ± 0.02 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        831.35 ± 1.35 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         47.07 ± 1.41 |

## cap185-2: llama-server tp4 -np 8 -c 262144, server-bench conc 1,8
# qwen38-27b-q8_0-powercap-cap185-2-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T15:22:52
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 62.5 | 32.8 | 199 | 7.7 | 46.2 | 1.83 | 7.8 |
| 8 | 16 | 62.6 | 65.5 | 398 | 15.3 | 12.7 | 6.47 | 31.2 |

## cap175-3: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        222.27 ± 4.28 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         19.65 ± 0.01 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        817.56 ± 1.40 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         47.21 ± 1.38 |

## cap175-3: llama-server tp4 -np 8 -c 262144, server-bench conc 1,8
# qwen38-27b-q8_0-powercap-cap175-3-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T15:27:31
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 63.4 | 32.3 | 196 | 7.6 | 46.2 | 1.86 | 7.9 |
| 8 | 16 | 63.5 | 64.5 | 392 | 15.1 | 12.5 | 6.55 | 31.7 |

## cap160-4: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        215.14 ± 4.14 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         19.01 ± 0.02 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        793.21 ± 1.26 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         46.46 ± 1.32 |

## cap160-4: llama-server tp4 -np 8 -c 262144, server-bench conc 1,8
# qwen38-27b-q8_0-powercap-cap160-4-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T15:32:54
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 63.5 | 32.3 | 196 | 7.6 | 45.6 | 1.90 | 7.9 |
| 8 | 16 | 65.0 | 63.0 | 383 | 14.8 | 12.2 | 6.69 | 32.4 |

## cap150-5: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        209.61 ± 4.29 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         18.47 ± 0.01 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        775.61 ± 0.42 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         45.77 ± 1.31 |

## cap150-5: llama-server tp4 -np 8 -c 262144, server-bench conc 1,8
# qwen38-27b-q8_0-powercap-cap150-5-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T15:37:56
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 64.0 | 32.0 | 195 | 7.5 | 45.2 | 1.95 | 8.0 |
| 8 | 16 | 66.5 | 61.6 | 375 | 14.4 | 11.9 | 6.84 | 33.2 |

## cap200-6: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |       226.69 ± 10.57 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         20.28 ± 0.02 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |        847.68 ± 1.64 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         47.11 ± 1.38 |

## cap200-6: llama-server tp4 -np 8 -c 262144, server-bench conc 1,8
# qwen38-27b-q8_0-powercap-cap200-6-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T15:43:11
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 62.0 | 33.0 | 201 | 7.7 | 46.1 | 1.81 | 7.8 |
| 8 | 16 | 62.3 | 65.7 | 399 | 15.4 | 13.1 | 6.36 | 31.1 |
# done 2026-09-07T15:45:27+00:00
