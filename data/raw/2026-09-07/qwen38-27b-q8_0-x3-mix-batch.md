# qwen38-27b-q8_0-x3: 1 x 128K + 7 x 4K at once, per-slot 8 x 128K server, batch/micro-batch 2048 (extras2) vs 1024 vs 512  2026-09-07T02:15:13+00:00
| server | mix | wall s | long TTFT s | long gen t/s | long wall s | short TTFT mean / max s | short gen t/s | short wall mean / max s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| per-slot 8 x 128K, batch 1024 | 1 × 130,816 + 7 × 4,096 | 337 | 282 | 16.8 | 337 | 15.4 / 45.7 | 0.8 | 332 / 333 |
| per-slot 8 x 128K, batch 512 | 1 × 130,816 + 7 × 4,096 | 365 | 311 | 35.8 | 365 | 15.3 / 50.2 | 0.9 | 319 / 356 |
