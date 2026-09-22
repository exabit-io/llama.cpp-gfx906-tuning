patch:            mxxm-substrate | mx-llama.cpp @ 0c81bd502, 157 commits squash-merged onto v0.4.1 | GROUP verdict | the fork substrate as a whole
axis:             multi-user
zero point:       build-stock-v041-rccl (stock upstream v0.4.1, RCCL on) measured 2026-09-22
recipe:           4x64K and 1x254K | q8_0 K | 125 W/die | --cache-ram 49152 | -ngl all | RCCL + gated custom AR
metric:           decode tok/s/slot and prefill t/s at the design point; full precision
result:           4x64K: decode +14.23%, prefill +24.86% over stock (n=4 per arm)
effect:           decode +14.23%, prefill +24.86%
stats:            all contrasts p=0.0286-0.0571, q=0.0686 PASS | BH over m=12 | n=4 fresh per arm
evidence:         group-level
structural:       standalone
verdict:          improves
bin:              both
would change if:  attribution to individual commits changing the picture — a subset may carry all of it, and some members may regress
notes:            THIS IS A GROUP VERDICT AND CANNOT SUBSTITUTE FOR BINNING ITS MEMBERS. The substrate is one
                  squashed commit, so no member can be left out of it; 53 of its 138 code commits
                  reverse-apply as discrete delta-minus units and the other 85 are separable only as 12
                  feature groups. None is screened yet -- the first attempt (screen.tsv) was invalid.
                  Re-measured on the campaign build config after an earlier three-way compared arms whose
                  FA_QUANTS differed.
