# qwen38-27b-q8_0-x2 M: MTP on the 4-die tensor split, single stream unless noted  2026-09-06T22:18:07+00:00
## M1 draft length (greedy, 400 tokens)
| variant | prompt | prompt tok | gen tok | gen t/s | TTFT s | draft acceptance (server log) |
|---|---|---:|---:|---:|---:|---|
| none | free | 42 | 400 | 44.27 | 0.34 | - |
| none | edit | 1789 | 400 | 44.24 | 2.34 | - |
| mtp-n1 | free | 42 | 400 | 56.56 | 0.39 | slot print_timing: id  0 | task 5 | draft acceptance = 0.86854 (  185 accepted /   213 gen |
| mtp-n1 | edit | 1789 | 400 | 59.01 | 2.49 | slot print_timing: id  0 | task 227 | draft acceptance = 0.95098 (  194 accepted /   204 g |
| mtp-n2 | free | 42 | 400 | 64.76 | 0.39 | slot print_timing: id  0 | task 5 | draft acceptance = 0.75157 (  239 accepted /   318 gen |
| mtp-n2 | edit | 1789 | 400 | 72.98 | 2.51 | slot print_timing: id  0 | task 173 | draft acceptance = 0.90813 (  257 accepted /   283 g |
| mtp-n3 | free | 42 | 400 | 65.83 | 0.39 | slot print_timing: id  0 | task 5 | draft acceptance = 0.63971 (  261 accepted /   408 gen |
| mtp-n3 | edit | 1789 | 400 | 83.34 | 2.50 | slot print_timing: id  0 | task 151 | draft acceptance = 0.88379 (  289 accepted /   327 g |
| mtp-n4 | free | 42 | 400 | 65.22 | 0.40 | slot print_timing: id  0 | task 5 | draft acceptance = 0.54819 (  273 accepted /   498 gen |
| mtp-n4 | edit | 1789 | 400 | 89.11 | 2.56 | slot print_timing: id  0 | task 139 | draft acceptance = 0.84341 (  307 accepted /   364 g |
| mtp-n6 | free | 42 | 400 | 57.14 | 0.40 | slot print_timing: id  0 | task 5 | draft acceptance = 0.41134 (  283 accepted /   688 gen |
| mtp-n6 | edit | 1789 | 400 | 92.87 | 2.52 | slot print_timing: id  0 | task 129 | draft acceptance = 0.79469 (  329 accepted /   414 g |
| mtp-n8 | free | 42 | 400 | 32.08 | 0.41 | slot print_timing: id  0 | task 5 | draft acceptance = 0.32466 (  287 accepted /   884 gen |
| mtp-n8 | edit | 1789 | 400 | 58.14 | 3.29 | slot print_timing: id  0 | task 125 | draft acceptance = 0.70270 (  338 accepted /   481 g |
## M2 sampling: the model's defaults (temp 1.0, top-k 20, top-p 0.95) and temp 0.7 / top-p 0.95
| variant | prompt | prompt tok | gen tok | gen t/s | TTFT s | draft acceptance (server log) |
|---|---|---:|---:|---:|---:|---|
| none-t1.0 | free | 42 | 400 | 44.94 | 1.18 | - |
| none-t1.0 | edit | 1789 | 400 | 44.85 | 2.40 | - |
| mtp-t1.0 | free | 42 | 400 | 64.70 | 0.39 | slot print_timing: id  0 | task 5 | draft acceptance = 0.59302 (  255 accepted /   430 gen |
| mtp-t1.0 | edit | 1789 | 400 | 86.02 | 2.49 | slot print_timing: id  0 | task 157 | draft acceptance = 0.88685 (  290 accepted /   327 g |
| none-t0.7 | free | 42 | 400 | 40.46 | 0.67 | - |
| none-t0.7 | edit | 1789 | 400 | 40.51 | 2.41 | - |
| mtp-t0.7 | free | 42 | 400 | 65.66 | 0.39 | slot print_timing: id  0 | task 5 | draft acceptance = 0.76099 (  277 accepted /   364 gen |
| mtp-t0.7 | edit | 1789 | 400 | 77.85 | 2.50 | slot print_timing: id  0 | task 135 | draft acceptance = 0.97059 (  297 accepted /   306 g |
## M3 depth: summarise 32K and 128K tokens of wikitext (greedy, 400 tokens)
| variant | prompt | prompt tok | gen tok | gen t/s | TTFT s | draft acceptance (server log) |
|---|---|---:|---:|---:|---:|---|
| none-32k | summary@32768 | 32787 | 400 | 40.45 | 45.52 | - |
| mtp-32k | summary@32768 | 32787 | 400 | 61.40 | 47.98 | slot print_timing: id  0 | task 5 | draft acceptance = 0.66085 (  265 accepted /   401 gen |
| none-128k | summary@131072 | 131091 | 400 | 35.24 | 258.39 | - |
| mtp-128k | summary@131072 | 131091 | 400 | 60.20 | 272.34 | slot print_timing: id  0 | task 5 | draft acceptance = 0.89231 (  290 accepted /   325 gen |
## M4 concurrency: eight realistic prompts, 300 tokens, greedy, 8 requests per level (16 at the 16-slot level); the verify batch per step is slots x (draft+1)
| variant | conc | reqs | wall s | agg gen t/s | per-req gen t/s | TTFT s | accepted / drafted (server log) |
|---|---:|---:|---:|---:|---:|---:|---|
| none (np 8) | 1 | 8 | 58.3 | 41.2 | 45.0 | 0.62 | - |
| none (np 8) | 2 | 8 | 50.6 | 47.4 | 29.6 | 1.12 | - |
| none (np 8) | 4 | 8 | 41.9 | 57.3 | 19.3 | 2.28 | - |
| none (np 8) | 8 | 8 | 22.0 | 109.2 | 17.7 | 4.29 | - |
| mtp-n1 (np 8) | 1 | 8 | 46.8 | 51.2 | 58.1 | 0.68 | 1120 / 1268 = 0.88 |
| mtp-n1 (np 8) | 2 | 8 | 39.8 | 60.3 | 37.5 | 1.06 | 1119 / 1271 = 0.88 |
| mtp-n1 (np 8) | 4 | 8 | 35.7 | 67.3 | 21.4 | 2.09 | 1118 / 1270 = 0.88 |
| mtp-n1 (np 8) | 8 | 8 | 27.6 | 86.8 | 14.4 | 4.59 | 1114 / 1274 = 0.87 |
| mtp-n2 (np 8) | 1 | 8 | 40.8 | 58.9 | 68.4 | 0.68 | 1461 / 1850 = 0.79 |
| mtp-n2 (np 8) | 2 | 8 | 37.1 | 64.8 | 41.1 | 1.14 | 1470 / 1835 = 0.80 |
| mtp-n2 (np 8) | 4 | 8 | 41.8 | 57.4 | 17.5 | 2.16 | 1462 / 1848 = 0.79 |
| mtp-n2 (np 8) | 8 | 8 | 29.2 | 82.1 | 14.1 | 4.63 | 1452 / 1869 = 0.78 |
| mtp-n3 (np 8) | 1 | 8 | 38.5 | 62.4 | 73.9 | 0.68 | 1622 / 2296 = 0.71 |
| mtp-n3 (np 8) | 2 | 8 | 36.3 | 66.0 | 42.3 | 1.10 | 1629 / 2273 = 0.72 |
| mtp-n3 (np 8) | 4 | 8 | 39.4 | 60.9 | 19.4 | 2.15 | 1606 / 2338 = 0.69 |
| mtp-n3 (np 8) | 8 | 8 | 29.9 | 80.2 | 14.3 | 4.34 | 1598 / 2366 = 0.68 |
| none (np 16) | 16 | 16 | 44.6 | 107.5 | 8.5 | 7.20 | - |
| mtp-n1 (np 16) | 16 | 16 | 46.7 | 102.9 | 8.6 | 7.66 | 2227 / 2548 = 0.87 |
| mtp-n3 (np 16) | 16 | 16 | 51.3 | 93.5 | 8.6 | 7.77 | 3219 / 4668 = 0.69 |
# done 2026-09-06T22:55:24+00:00
