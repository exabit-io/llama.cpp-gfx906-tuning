# qwen38-27b-gfx906-master-validate  2026-09-08T14:50:24+00:00  fork b10912 + our series vs production (fork b10254 + series), fork pristine b10912, upstream pristine b10859

## A. test-backend-ops (ROCm0): MUL_MAT:   1288/1288 tests passed ; MUL_MAT_ID: ?; RMS_NORM:   51/51 tests passed; GATED_DELTA_NET:   36/36 tests passed

## B. perplexity 16K, new build: Final estimate: PPL = 5.6173 +/- 0.06235 (production 5.5969)

## C. greedy vs production differs at char 199 (-1 identical); KL at 64-token batches: chunk PPL ln(PPL(Q)/PPL(base)) KL Divergence Δp RMS Same top p;Mean ln(PPL(Q)/PPL(base)) : 0.003164 ± 0.001801;Mean KLD: 0.003627 ± 0.001156;90.0% KLD: 0.001215;90.0% Δp: 0.495%;Same top p: 98.925 ± 0.114 %;

## D. llama-bench -p 2048 -n 128 -r 3 (tp4 + rocm0); custom AR env on the two builds that carry the gate
| build | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |
|---|---:|---:|---:|---:|
| prod-b10254 | 1126.17 ± 1.22 | 57.95 ± 1.82 | 331.25 ± 0.08 | 21.74 ± 0.04 | 
| new-gfx906-master | 1360.63 ± 3.06 | 42.77 ± 13.77 | 423.95 ± 0.12 | 20.42 ± 1.76 | 
| fork-b10912-pristine | 1364.29 ± 2.84 | 41.27 ± 7.84 | 424.87 ± 0.15 | 21.48 ± 0.74 | 
| upstream-master-pristine | 843.25 ± 1.19 | 46.07 ± 2.20 | 233.86 ± 0.10 | 20.33 ± 0.02 | 
| new-gfx906-master-noar | 1360.58 ± 4.62 | 40.10 ± 10.75 | 424.02 ± 0.03 | 19.13 ± 3.09 | 
| prod-b10254-repeat | 1128.41 ± 1.05 | 58.15 ± 1.81 | 331.74 ± 0.06 | 21.73 ± 0.05 | 
| new-gfx906-master-repeat | 1362.37 ± 2.66 | 44.13 ± 13.06 | 424.18 ± 0.07 | 18.21 ± 3.77 | 

## E. batched-bench decode tok/s at 2K, tp4 -npl 1..17,24,32 (MMVQ 1-16, MMQ 17+) and rocm0 -npl 1,2,4,8 at 512
### llama.cpp-gfx906-master tp4
1:25.82 2:39.82 3:114.44 4:127.22 5:120.41 6:121.26 7:166.18 8:174.35 9:113.13 10:124.90 11:132.52 12:141.40 13:149.67 14:157.34 15:162.23 16:169.22 17:165.98 24:219.67 32:258.60 
### llama.cpp-gfx906-master rocm0
1:22.41 2:37.58 4:56.22 8:68.84 
### llama.cpp-prod tp4
1:53.48 2:91.34 3:122.74 4:127.68 5:146.66 6:154.17 7:164.88 8:175.09 9:183.52 10:187.35 11:193.00 12:197.16 13:197.21 14:200.11 15:201.94 16:202.68 17:153.33 24:191.85 32:214.84 
### llama.cpp-prod rocm0
1:20.60 2:37.56 4:53.55 8:70.32 

## F. single user MTP draft 3 on the new build: LLAMA_ENABLE_MTP_OPT off / on (decode-only wave tok/s)
| flag | depth | per-req gen t/s | accepted / drafted |
|---|---:|---:|---|
| mtpopt-off | 2048 | 76.0 | 204 / 284 = 0.72 |
| mtpopt-on | 2048 | 76.3 | 204 / 284 = 0.72 |
| mtpopt-off | 32768 | 65.8 | 200 / 295 = 0.68 |
| mtpopt-on | 32768 | 66.4 | 200 / 295 = 0.68 |
# done 2026-09-08T15:38:31+00:00
