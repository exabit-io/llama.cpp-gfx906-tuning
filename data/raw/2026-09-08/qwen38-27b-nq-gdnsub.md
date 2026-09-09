# qwen38-27b-nq-gdnsub  2026-09-08T11:21:15+00:00  GDN fold sub-masks vs production (restricted fusion build)
| mask | meaning | greedy differs at char | PPL (ref 5.5969) |
|---|---|---:|---|
| 1 | q/k-l2-only | 556 | 5.5969 +/- 0.06197 |
| 2 | beta-sigmoid-only | -1 | 5.5969 +/- 0.06197 |
| 7 | all | 556 | 5.5969 +/- 0.06197 |
| 0 | off | -1 | 5.5969 +/- 0.06197 |
# done 2026-09-08T11:37:17+00:00
