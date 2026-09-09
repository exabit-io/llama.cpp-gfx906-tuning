# qwen38-27b-q8_0-opt E: llama-server 4 dies tensor, -np 1 -c 16384, greedy, n_predict 400; two prompts (free generation, code edit)  2026-09-06T19:25:34+00:00
| variant | prompt | prompt tok | gen tok | gen t/s | TTFT s | draft acceptance (server log) |
|---|---|---:|---:|---:|---:|---|
| none | free | 42 | 400 | 44.86 | 0.78 | - |
| none | edit | 1789 | 400 | 44.81 | 2.49 | - |
| mtp | free | 42 | 400 | 62.51 | 0.40 | slot print_timing: id  0 | task 5 | draft acceptance = 0.63971 (  261 accepted /   408 gen |
| mtp | edit | 1789 | 400 | 84.81 | 2.51 | slot print_timing: id  0 | task 151 | draft acceptance = 0.88379 (  289 accepted /   327 g |
| mtp-n2 | free | 42 | 400 | 65.75 | 0.39 | slot print_timing: id  0 | task 5 | draft acceptance = 0.75157 (  239 accepted /   318 gen |
| mtp-n2 | edit | 1789 | 400 | 74.09 | 2.53 | slot print_timing: id  0 | task 173 | draft acceptance = 0.90813 (  257 accepted /   283 g |
| ngram-mod | free | 42 | 400 | 44.80 | 0.34 | - |
| ngram-mod | edit | 1789 | 400 | 44.78 | 2.38 | slot print_timing: id  0 | task 412 | draft acceptance = 0.59242 (  125 accepted /   211 g |
| draft-0.8b | free | 42 | 400 | 11.62 | 0.49 | slot print_timing: id  0 | task 5 | draft acceptance = 0.27803 (  186 accepted /   669 gen |
| draft-0.8b | edit | 1789 | 400 | 24.06 | 2.81 | slot print_timing: id  0 | task 226 | draft acceptance = 0.69484 (  296 accepted /   426 g |
| draft-2b | free | 42 | 400 | 15.91 | 0.79 | slot print_timing: id  0 | task 5 | draft acceptance = 0.47126 (  246 accepted /   522 gen |
| draft-2b | edit | 1789 | 400 | 25.51 | 2.93 | slot print_timing: id  0 | task 166 | draft acceptance = 0.74328 (  304 accepted /   409 g |
# done 2026-09-06T19:33:04+00:00
