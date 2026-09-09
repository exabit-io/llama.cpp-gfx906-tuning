# qwen38-27b-q8_0-mixed2  2026-09-07T04:23:44+00:00  one 130,816-token prompt beside seven 4,096-token prompts, 256 generated each, all sent at once
| layout | wave | wall s | long TTFT s | long gen t/s | long wall s | short TTFT mean / max s | short gen t/s | short wall mean / max s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A: 2 x tp2 (long -> tp2 np2, short -> tp2 np8) | 1 x 130,816 + 7 x 4,096 | 487 | 476 | 23.1 | 487 | 29.2 / 70.0 | 8.1 | 90 / 90 |
| B: tp3 + 1 die (long -> tp3 np4, short -> rocm3 np8) | client failed: urllib.error.HTTPError: HTTP Error 400: Bad Request | | | | | | | |
| control: one tp4 np8 8x128K server takes both | 1 x 130,816 + 7 x 4,096 | 320 | 262 | 13.0 | 320 | 17.7 / 44.0 | 0.9 | 318 / 319 |
# done 2026-09-07T04:46:23+00:00
| B (rerun): tp3 + 1 die (long -> tp3 np4, short -> rocm3 np6 x 4608) | 1 x 130,816 + 7 x 4,096 | 412 | 401 | 25.5 | 412 | 42.5 / 99.0 | 5.9 | 155 / 170 |
# B rerun done 2026-09-07T11:17:52+00:00
