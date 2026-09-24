#!/usr/bin/env python3
"""binstats.py — bin the Exabit patchsets from binrun.tsv. Written BEFORE any binrun data existed.

Pre-registered rule (RESURVEY-ACTION-PLAN §1.10, lead-approved 2026-09-20):
  contrast   arm vs its reference: base, except gdn-producer-fold vs norm-add-fusion and
             mmvq-batch1-knobs = the whole mmvq patchset vs base (amended, see REF)
  statistic  run-level: multi-user MEAN of n=5; single-user MEDIAN of n=5 (heavy-tailed stalls, §Part 0)
  test       two-sided exact permutation, 5 vs 5 = 252 arrangements (p floor 0.0079)
  family     6 patchsets x 2 axes x 2 metrics (decode, prefill) = 24 tests (amended from 28, see REF), BH FDR q = 0.10
  per axis   improves  = some metric q<0.10 AND effect >= +2%, and no metric q<0.10 AND <= -2%
             regresses = some metric q<0.10 AND effect <= -2%, and none improves
             mixed     = one improves, one regresses  -> reported, never auto-binned
             neutral   = otherwise
  bin        improves on both -> both; on multi only -> multi-user-only; on single only -> single-user-only
             regresses on both -> regresses-both; neither -> neutral-drop
             (R2.7: a regression on one axis never blocks the other axis's build)
usage: binstats.py binrun.tsv
"""
import sys, itertools, statistics as st

# AMENDED 2026-09-24 00:05, BEFORE any cell was measured: term 01 does not compile without term 05 (its
# c4-series resolution uses q8_fast, which 05 declares), so mmvq-16col cannot exist as an arm. The arm
# 'mmvq-batch1-knobs' (01 15 05 09) is the whole inseparable mmvq patchset and is binned against base.
# Family: 6 patchsets x 2 axes x 2 metrics = 24.
LABEL = {'mmvq-batch1-knobs': 'mmvq-q8-fastpath(01,05,09,15)'}
REF = {'norm-add-fusion': 'base', 'gdn-producer-fold': 'norm-add-fusion',
       'mmvq-batch1-knobs': 'base', 's1b-repacked-matvec': 'base', 'fa-head256-rows': 'base',
       'dpp-warp-reductions': 'base'}
AXES = ('multi', 'single'); METRICS = ('decode', 'prefill'); Q = 0.10; FLOOR = 2.0

data = {}
for line in open(sys.argv[1]):
    f = line.rstrip('\n').split('\t')
    if len(f) < 5:
        continue
    try:
        d, p = float(f[3]), float(f[4])
    except ValueError:
        print(f"non-numeric row, not used: {line.strip()}"); continue
    data.setdefault((f[0], f[1]), []).append((d, p))

def stat(axis, xs):
    return st.mean(xs) if axis == 'multi' else st.median(xs)

def perm_p(axis, a, b):
    obs = abs(stat(axis, b) - stat(axis, a)); pool = a + b; n = len(a); hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        s = set(idx)
        x = [pool[i] for i in s]; y = [pool[i] for i in range(len(pool)) if i not in s]
        tot += 1; hits += abs(stat(axis, y) - stat(axis, x)) >= obs - 1e-12
    return hits / tot

tests = []
for arm, ref in REF.items():
    for axis in AXES:
        A = data.get((axis, ref), []); B = data.get((axis, arm), [])
        for mi, metric in enumerate(METRICS):
            a = [r[mi] for r in A]; b = [r[mi] for r in B]
            if len(a) < 5 or len(b) < 5:
                tests.append(dict(arm=arm, ref=ref, axis=axis, metric=metric, n=(len(a), len(b)), eff=None, p=None)); continue
            eff = 100 * (stat(axis, b) / stat(axis, a) - 1)
            tests.append(dict(arm=arm, ref=ref, axis=axis, metric=metric, n=(len(a), len(b)), eff=eff, p=perm_p(axis, a, b),
                              ra=stat(axis, a), rb=stat(axis, b)))

# BH over the whole declared family of 28; an untested member counts as p=1 so the family is never shrunk
m = len(REF) * len(AXES) * len(METRICS)
ps = sorted(((t['p'] if t['p'] is not None else 1.0), i) for i, t in enumerate(tests))
qs = [0.0] * len(tests); prev = 1.0
for rank in range(len(ps), 0, -1):
    p, i = ps[rank - 1]; prev = min(prev, p * m / rank); qs[i] = prev
for t, q in zip(tests, qs):
    t['q'] = q

print(f"{'patchset':22} {'axis':6} {'metric':8} {'ref':16} {'n':5} {'ref val':>9} {'arm val':>9} {'effect':>8} {'p':>7} {'q':>7}")
for t in tests:
    if t['eff'] is None:
        print(f"{t['arm']:22} {t['axis']:6} {t['metric']:8} {t['ref']:16} {str(t['n']):5}  INCOMPLETE"); continue
    print(f"{t['arm']:22} {t['axis']:6} {t['metric']:8} {t['ref']:16} {t['n'][1]:<5} {t['ra']:9.3f} {t['rb']:9.3f} "
          f"{t['eff']:+7.2f}% {t['p']:7.4f} {t['q']:7.4f}")

print("\nBINS")
for arm in REF:
    ax = {}
    for axis in AXES:
        ts = [t for t in tests if t['arm'] == arm and t['axis'] == axis]
        if any(t['eff'] is None for t in ts):
            ax[axis] = 'incomplete'; continue
        up = any(t['q'] < Q and t['eff'] >= FLOOR for t in ts)
        dn = any(t['q'] < Q and t['eff'] <= -FLOOR for t in ts)
        ax[axis] = 'mixed' if up and dn else 'improves' if up else 'regresses' if dn else 'neutral'
    if 'incomplete' in ax.values() or 'mixed' in ax.values():
        b = 'NOT BINNED (' + ', '.join(f'{k}={v}' for k, v in ax.items()) + ')'
    elif ax['multi'] == 'improves' and ax['single'] == 'improves': b = 'both'
    elif ax['multi'] == 'improves': b = 'multi-user-only'
    elif ax['single'] == 'improves': b = 'single-user-only'
    elif ax['multi'] == 'regresses' and ax['single'] == 'regresses': b = 'regresses-both'
    else: b = 'neutral-drop'
    print(f"  {LABEL.get(arm, arm):30} multi={ax['multi']:10} single={ax['single']:10} -> {b}")
