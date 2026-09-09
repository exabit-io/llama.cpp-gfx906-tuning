# round 2, rocm0, Q4_K_M  2026-09-03T16:15:21+00:00
## llama-bench pp1300 / tg256@d1300, fa on, ub 2048
|        121.09 ± 0.00 ||
|         17.28 ± 0.01 ||
## llama-batched-bench npp1300 ntg256 npl1, fa on
|  1300 |    256 |    1 |   1556 |   10.621 |   122.40 |   14.892 |    17.19 |   25.513 |    60.99 |
## llama-completion, ~1300-token prompt, -n 256, greedy
0.28.852.806 I common_perf_print: prompt eval time =    9185.42 ms /  1142 tokens (    8.04 ms per token,   124.33 tokens per second)
0.28.852.808 I common_perf_print:        eval time =   14742.83 ms /   255 runs   (   57.82 ms per token,    17.30 tokens per second)
0.28.852.808 I common_perf_print:       total time =   24059.25 ms /  1397 tokens
# done 2026-09-03T16:17:38+00:00
