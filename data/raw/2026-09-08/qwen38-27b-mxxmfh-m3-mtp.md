# qwen38-27b-mxxmfh-m3-mtp  2026-09-08T10:19:48+00:00  production build tp4, greedy, 300 generated, N concurrent distinct wikitext prompts, two waves (wave 2 = clean decode); libggml-hip: /opt/llama.cpp-mxxm-fh/lib/libggml-hip.so.0
| variant | depth | conc | wall s | prompt tok | TTFT mean / max s | per-req gen t/s | sum per-req gen t/s | wave agg gen t/s | accepted / drafted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| none wave 1 (prefill + decode) | 2048 | 4 | 20 | 2067 | 7 / 9 | 25.1 | 100.3 | 59.9 | - |
| none wave 2 (decode only) | 2048 | 4 | 12 | 4 | 1 / 1 | 29.2 | 116.8 | 96.2 | - |
| mtp-n1 wave 1 (prefill + decode) | 2048 | 4 | 20 | 2067 | 8 / 11 | 26.7 | 106.8 | 59.4 | 544 / 650 = 0.84 |
| mtp-n1 wave 2 (decode only) | 2048 | 4 | 12 | 4 | 1 / 1 | 33.2 | 132.8 | 103.3 | 545 / 648 = 0.84 |
| mtp-n2 wave 1 (prefill + decode) | 2048 | 4 | 21 | 2067 | 8 / 11 | 26.3 | 105.0 | 57.6 | 714 / 961 = 0.74 |
| mtp-n2 wave 2 (decode only) | 2048 | 4 | 12 | 4 | 1 / 1 | 33.5 | 134.0 | 103.3 | 704 / 981 = 0.72 |
| mtp-n3 wave 1 (prefill + decode) | 2048 | 4 | 21 | 2067 | 8 / 11 | 26.6 | 106.3 | 58.2 | 786 / 1224 = 0.64 |
| mtp-n3 wave 2 (decode only) | 2048 | 4 | 12 | 4 | 1 / 1 | 33.3 | 133.0 | 102.1 | 786 / 1224 = 0.64 |
| none wave 1 (prefill + decode) | 2048 | 8 | 36 | 2067 | 13 / 19 | 15.4 | 123.4 | 67.3 | - |
| none wave 2 (decode only) | 2048 | 8 | 20 | 4 | 1 / 2 | 20.1 | 161.1 | 120.4 | - |
| mtp-n1 wave 1 (prefill + decode) | 2048 | 8 | 37 | 2067 | 15 / 21 | 14.8 | 118.1 | 64.6 | 1097 / 1294 = 0.85 |
| mtp-n1 wave 2 (decode only) | 2048 | 8 | 21 | 4 | 1 / 2 | 19.7 | 157.6 | 113.5 | 1103 / 1286 = 0.86 |
| none wave 1 (prefill + decode) | 32768 | 4 | 160 | 32787 | 44 / 51 | 9.8 | 39.2 | 7.5 | - |
| none wave 2 (decode only) | 32768 | 4 | 21 | 4 | 1 / 1 | 25.1 | 100.6 | 57.3 | - |
| mtp-n1 wave 1 (prefill + decode) | 32768 | 4 | 174 | 32787 | 47 / 50 | 10.7 | 43.0 | 6.9 | 560 / 635 = 0.88 |
| mtp-n1 wave 2 (decode only) | 32768 | 4 | 25 | 4 | 1 / 2 | 29.1 | 116.4 | 48.2 | 561 / 634 = 0.88 |
| mtp-n2 wave 1 (prefill + decode) | 32768 | 4 | 176 | 32787 | 48 / 53 | 11.4 | 45.5 | 6.8 | 735 / 915 = 0.80 |
| mtp-n2 wave 2 (decode only) | 32768 | 4 | 25 | 4 | 1 / 2 | 30.4 | 121.6 | 48.7 | 726 / 933 = 0.78 |
| mtp-n3 wave 1 (prefill + decode) | 32768 | 4 | 175 | 32787 | 48 / 54 | 12.4 | 49.6 | 6.9 | 819 / 1125 = 0.73 |
| mtp-n3 wave 2 (decode only) | 32768 | 4 | 25 | 4 | 1 / 2 | 30.9 | 123.6 | 48.8 | 819 / 1120 = 0.73 |
| none wave 1 (prefill + decode) | 32768 | 8 | 309 | 32787 | 44 / 51 | 4.6 | 36.7 | 7.8 | - |
| none wave 2 (decode only) | 32768 | 8 | 89 | 4 | 1 / 2 | 16.6 | 133.0 | 26.9 | - |
| mtp-n1 wave 1 (prefill + decode) | 32768 | 8 | 348 | 32787 | 50 / 55 | 4.9 | 39.2 | 6.9 | 1120 / 1269 = 0.88 |
| mtp-n1 wave 2 (decode only) | 32768 | 8 | 103 | 4 | 2 / 4 | 17.5 | 140.0 | 23.2 | 1126 / 1263 = 0.89 |
# done 2026-09-08T11:03:18+00:00
