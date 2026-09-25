# Round 1b — stacked round-1 patchsets (pre-registered 2026-09-25, before any round-1b data)

Question (lead, 2026-09-25): do the round-1 patchsets that are individually real but below the 2% floor clear it
when combined? Selection uses round-1 data, so the test is on FRESH runs only; round-1 numbers are the prediction.

Arms (built on master a23e12438, commits from gfx906-candidates, binrun-arms-r1b.txt):
- combo-both  = norm-add-fusion (02 03) + gdn-producer-fold (04 07 08 10 12) + dpp-warp-reductions (23 28) + max-ilp flag:
  every round-1 patchset that was significant and non-negative on both axes' decode.
- combo-multi = combo-both + mmvq-q8-fastpath (01 05 09 15): mmvq was +0.57% multi decode but -1.19% single decode,
  so it is a multi-user-only candidate. Excluded: s1b-repacked-matvec (|effect| <= 0.2%), fa-head256-rows (regresses-both).

Rule (unchanged from round 1 except n): binstats.py with binstats-r1b.json; multi-user mean of n=5, single-user median
of n=6 (lead 2026-09-25; median floor 6v6 = 0.013); two-sided exact permutation; BH q<0.10 over the family
2 arms x 2 axes x 2 metrics = 8; improves = q<0.10 AND >= +2%. Bins as in round 1 (both / multi-user-only / ...).

Additive prediction (product of the round-1 effects, i.e. no interaction). 'Synergy' = measured minus predicted;
descriptive only, not a bin criterion:

| arm | 4x64K decode | 4x64K prefill | 1x255K decode | 1x255K prefill |
|---|---:|---:|---:|---:|
| combo-both | +3.93% | +1.52% | +2.51% | +0.88% |
| combo-multi | +4.53% | +1.49% | +1.29% | +0.90% |

Run: ROUND=r1b ARMS_FILE=binrun-arms-r1b.txt NMULTI=5 NSINGLE=6 binrun.sh, then fncompat.sh (Flash-Next gate), chained by PID.
