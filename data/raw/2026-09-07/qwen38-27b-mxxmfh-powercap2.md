# qwen38-27b-mxxmfh-powercap2  2026-09-07T21:38:12+00:00  GPU power-cap sweep 2, Qwen3.8-27B Q8_0, production build /opt/llama.cpp-mxxm-fh
Per point: llama-bench rocm0 + tp4 (-p 2048 -n 256 -d 0 -r 2), then llama-server tp4 -np 16 -c 524288 with server-bench conc 8,16 (1300-token prompts, 256 gen).
Order: 200, 180, 175, 170, 165, 160, 155, 200 (reference at both ends). Power per phase: slice qwen38-27b-mxxmfh-powercap2-clocks.txt (W per die, 5 s) and smc-power-*.log by the PHASE timestamps in qwen38-27b-mxxmfh-powercap2.progress.

## cap200-1: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        320.68 ± 0.02 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         20.29 ± 0.04 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |       1126.67 ± 1.80 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         48.86 ± 0.97 |

## cap200-1: llama-server tp4 -np 16 -c 524288, server-bench conc 8,16
# qwen38-27b-mxxmfh-powercap2-cap200-1-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T21:40:09
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 52.9 | 77.4 | 470 | 18.1 | 15.3 | 6.01 | 26.3 |
| 16 | 32 | 100.5 | 81.5 | 495 | 19.1 | 7.2 | 5.77 | 50.1 |

## cap180-2: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        309.99 ± 0.15 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         19.80 ± 0.04 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |       1095.57 ± 0.74 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         48.85 ± 0.85 |

## cap180-2: llama-server tp4 -np 16 -c 524288, server-bench conc 8,16
# qwen38-27b-mxxmfh-powercap2-cap180-2-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T21:45:11
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 53.7 | 76.3 | 464 | 17.9 | 14.7 | 5.93 | 26.7 |
| 16 | 32 | 103.1 | 79.5 | 483 | 18.6 | 7.0 | 5.91 | 51.4 |

## cap175-3: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        306.55 ± 0.02 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         19.69 ± 0.04 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |       1085.29 ± 1.57 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         48.55 ± 0.87 |

## cap175-3: llama-server tp4 -np 16 -c 524288, server-bench conc 8,16
# qwen38-27b-mxxmfh-powercap2-cap175-3-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T21:50:05
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 54.4 | 75.3 | 458 | 17.6 | 15.1 | 6.32 | 27.0 |
| 16 | 32 | 104.0 | 78.8 | 479 | 18.5 | 7.0 | 5.85 | 51.8 |

## cap170-4: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        303.62 ± 0.24 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         19.52 ± 0.03 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |       1073.17 ± 1.12 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         48.56 ± 0.82 |

## cap170-4: llama-server tp4 -np 16 -c 524288, server-bench conc 8,16
# qwen38-27b-mxxmfh-powercap2-cap170-4-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T21:55:14
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 54.5 | 75.1 | 457 | 17.6 | 15.0 | 6.32 | 27.1 |
| 16 | 32 | 104.3 | 78.5 | 477 | 18.4 | 7.0 | 5.87 | 52.0 |

## cap165-5: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        300.04 ± 0.20 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         19.36 ± 0.02 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |       1059.87 ± 1.28 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         48.20 ± 0.80 |

## cap165-5: llama-server tp4 -np 16 -c 524288, server-bench conc 8,16
# qwen38-27b-mxxmfh-powercap2-cap165-5-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T22:00:20
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 55.0 | 74.5 | 453 | 17.5 | 14.9 | 6.40 | 27.4 |
| 16 | 32 | 106.6 | 76.8 | 467 | 18.0 | 6.8 | 6.07 | 53.1 |

## cap160-6: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        295.41 ± 0.82 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         19.10 ± 0.03 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |       1046.02 ± 1.78 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         47.90 ± 0.76 |

## cap160-6: llama-server tp4 -np 16 -c 524288, server-bench conc 8,16
# qwen38-27b-mxxmfh-powercap2-cap160-6-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T22:05:41
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 55.9 | 73.3 | 446 | 17.2 | 14.8 | 6.57 | 27.8 |
| 16 | 32 | 106.4 | 77.0 | 468 | 18.0 | 6.8 | 5.99 | 53.0 |

## cap155-7: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        291.13 ± 1.06 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         18.85 ± 0.01 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |       1033.80 ± 1.38 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         47.43 ± 0.71 |

## cap155-7: llama-server tp4 -np 16 -c 524288, server-bench conc 8,16
# qwen38-27b-mxxmfh-powercap2-cap155-7-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T22:10:52
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 56.2 | 72.9 | 443 | 17.1 | 14.6 | 6.57 | 27.9 |
| 16 | 32 | 107.5 | 76.2 | 463 | 17.9 | 6.7 | 6.05 | 53.5 |

## cap200-8: llama-bench -p 2048 -n 256 -d 0 -r 2, rocm0 and tp4
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |          pp2048 |        321.54 ± 0.15 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0        |           tg256 |         20.76 ± 0.03 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |          pp2048 |       1129.04 ± 1.19 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       |  -1 |     2048 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg256 |         48.86 ± 0.86 |

## cap200-8: llama-server tp4 -np 16 -c 524288, server-bench conc 8,16
# qwen38-27b-mxxmfh-powercap2-cap200-8-server  prompt≈1300 tok  gen=256 tok (ignore_eos)  2026-09-07T22:16:14
| conc | reqs | wall s | agg gen t/s | total tok/s (pp+gen) | req/min | per-req gen t/s | TTFT s (prompt) | per-req wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 53.8 | 76.1 | 462 | 17.8 | 14.9 | 6.09 | 26.8 |
| 16 | 32 | 100.8 | 81.3 | 494 | 19.0 | 7.2 | 5.78 | 50.2 |
# done 2026-09-07T22:19:00+00:00
