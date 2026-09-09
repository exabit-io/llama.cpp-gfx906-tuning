# qwen38-27b-q8_0-mtpdepth2  2026-09-07T11:18:07+00:00  tp4, greedy, 300 generated, N concurrent distinct wikitext prompts, TWO waves: wave 2 re-sends the same prompts with cache_prompt on so every slot decodes together (clean decode factor)
| variant | depth | conc | wall s | prompt tok | TTFT mean / max s | per-req gen t/s | sum per-req gen t/s | wave agg gen t/s | accepted / drafted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| none wave 1 (prefill + decode) | 32768 | 4 | 197 | 32787 | 53 / 58 | 8.9 | 35.8 | 6.1 | - |
| none wave 2 (decode only) | 32768 | 4 | 21 | 4 | 1 / 1 | 24.2 | 96.9 | 58.3 | - |
| mtp-n1 wave 1 (prefill + decode) | 32768 | 4 | 208 | 32787 | 57 / 59 | 9.8 | 39.4 | 5.8 | 562 / 634 = 0.89 |
| mtp-n1 wave 2 (decode only) | 32768 | 4 | 25 | 4 | 1 / 2 | 26.9 | 107.6 | 47.5 | 560 / 636 = 0.88 |
| mtp-n2 wave 1 (prefill + decode) | 32768 | 4 | 212 | 32787 | 58 / 64 | 9.0 | 35.9 | 5.7 | 735 / 917 = 0.80 |
| mtp-n2 wave 2 (decode only) | 32768 | 4 | 28 | 4 | 1 / 2 | 21.2 | 84.8 | 42.4 | 734 / 917 = 0.80 |
| mtp-n2 wave 1 (prefill + decode) | 32768 | 2 | 104 | 32787 | 53 / 54 | 25.4 | 50.8 | 5.8 | 366 / 461 = 0.79 |
| mtp-n2 wave 2 (decode only) | 32768 | 2 | 11 | 4 | 1 / 1 | 44.6 | 89.3 | 55.6 | 361 / 472 = 0.76 |
| mtp-n3 wave 1 (prefill + decode) | 32768 | 2 | 105 | 32787 | 54 / 54 | 25.4 | 50.8 | 5.7 | 402 / 584 = 0.69 |
| mtp-n3 wave 2 (decode only) | 32768 | 2 | 11 | 4 | 1 / 1 | 45.8 | 91.7 | 56.9 | 402 / 584 = 0.69 |
| none wave 1 (prefill + decode) | 32768 | 2 | 101 | 32787 | 51 / 53 | 19.4 | 38.8 | 5.9 | - |
| none wave 2 (decode only) | 32768 | 2 | 12 | 4 | 0 / 0 | 32.5 | 65.1 | 49.1 | - |
| none wave 1 (prefill + decode) | 131072 | 4 | 1051 | 131091 | 269 / 276 | 5.8 | 23.3 | 1.1 | - |
| none wave 2 (decode only) | 131072 | 4 | 19 | 4 | 1 / 1 | 16.7 | 66.8 | 62.4 | - |
| mtp-n1 wave 1 (prefill + decode) | 131072 | 4 | 1129 | 131091 | 290 / 297 | 9.4 | 37.5 | 1.1 | 592 / 603 = 0.98 |
| mtp-n1 wave 2 (decode only) | 131072 | 4 | 19 | 4 | 3 / 4 | 20.1 | 80.4 | 61.7 | 591 / 604 = 0.98 |
| mtp-n2 wave 1 (prefill + decode) | 131072 | 4 | 1134 | 131091 | 292 / 300 | 12.0 | 48.1 | 1.1 | 779 / 833 = 0.94 |
| mtp-n2 wave 2 (decode only) | 131072 | 4 | 21 | 4 | 3 / 4 | 17.7 | 70.8 | 56.1 | 778 / 832 = 0.94 |
| mtp-n2 wave 1 (prefill + decode) | 131072 | 2 | 559 | 131091 | 283 / 290 | 22.6 | 45.3 | 1.1 | 389 / 418 = 0.93 |
| mtp-n2 wave 2 (decode only) | 131072 | 2 | 10 | 4 | 2 / 2 | 38.8 | 77.7 | 59.6 | 389 / 417 = 0.93 |
| mtp-n3 wave 1 (prefill + decode) | 131072 | 2 | 560 | 131091 | 283 / 292 | 26.2 | 52.5 | 1.1 | 433 / 491 = 0.88 |
| mtp-n3 wave 2 (decode only) | 131072 | 2 | 10 | 4 | 2 / 2 | 41.5 | 82.9 | 61.5 | 433 / 491 = 0.88 |
| none wave 1 (prefill + decode) | 131072 | 2 | 525 | 131091 | 265 / 271 | 14.1 | 28.2 | 1.1 | - |
| none wave 2 (decode only) | 131072 | 2 | 12 | 4 | 0 / 0 | 25.7 | 51.4 | 48.3 | - |
# done 2026-09-07T13:06:17+00:00
