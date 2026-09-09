# qwen38-27b-cap-phases  2026-09-08T14:36:30+00:00  production build tp4: decode-only (16 slots, 2K) and prefill-only (pp2048) per cap
| cap W | decode 16 slots tok/s | prefill pp2048 tok/s | mean die W during decode | during prefill |
|---:|---:|---:|---:|---:|
| 200 | 203.42 | 1129.98 ± 0.98 | see clocks.txt 14:36:32-14:37:50 | 14:37:50-14:38:36 |
| 170 | 194.09 | 1072.83 ± 0.97 | see clocks.txt 14:38:38-14:40:00 | 14:40:00-14:40:43 |
| 140 | 179.54 | 990.93 ± 1.15 | see clocks.txt 14:40:45-14:42:06 | 14:42:06-14:42:49 |
| 125 | 168.34 | 932.07 ± 0.32 | see clocks.txt 14:42:51-14:44:16 | 14:44:16-14:45:00 |
| 85 | 137.68 | 732.92 ± 0.63 | see clocks.txt 14:45:02-14:46:40 | 14:46:40-14:47:24 |
| 200 | 204.17 | 1130.14 ± 1.39 | see clocks.txt 14:47:26-14:48:41 | 14:48:41-14:49:20 |
# done 2026-09-08T14:49:22+00:00
