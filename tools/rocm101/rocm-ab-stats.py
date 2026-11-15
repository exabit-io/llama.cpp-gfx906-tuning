#!/usr/bin/env python3
"""rocm-ab-stats.py TSV [REF ARM] -- the survey's rule (night-20260919/binstats.py) for the two-arm ROCm A/B. Written before any
cell ran; the rule is in rocm-ab.sh's header.

  statistic  run-level: multi-user MEAN, single-user MEDIAN of the n runs per arm
  effect     r101 / r100 - 1
  test       two-sided exact permutation (5 vs 5 = 252 arrangements)
  family     2 axes x 2 metrics = 4 tests, Benjamini-Hochberg q < 0.10; counts only if also |effect| >= 2 %
Prints every run, the per-test line, and a verdict per axis once both metrics of that axis are complete.
"""
import itertools
import statistics as st
import sys

AXES = ("multi", "single"); METRICS = ("decode", "prefill"); Q = 0.10; FLOOR = 2.0
REF, ARM = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("r100", "r101")     # arm names in the TSV
data = {}
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    if len(f) < 5:
        continue
    try:
        d, p = float(f[3]), float(f[4])
    except ValueError:
        print(f"[stats] not used: {line.strip()}")
        continue
    data.setdefault((f[0], f[1]), []).append((d, p))


def stat(axis, xs):
    return st.mean(xs) if axis == "multi" else st.median(xs)


def perm_p(axis, a, b):
    obs = abs(stat(axis, b) - stat(axis, a)); pool = a + b; n = len(a); hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        s = set(idx)
        x = [pool[i] for i in s]; y = [pool[i] for i in range(len(pool)) if i not in s]
        tot += 1; hits += abs(stat(axis, y) - stat(axis, x)) >= obs - 1e-12
    return hits / tot


tests = []
for axis in AXES:
    a, b = data.get((axis, REF), []), data.get((axis, ARM), [])
    if not a or not b:
        continue
    for mi, metric in enumerate(METRICS):
        xa, xb = [r[mi] for r in a], [r[mi] for r in b]
        eff = (stat(axis, xb) / stat(axis, xa) - 1) * 100
        p = perm_p(axis, xa, xb) if len(xa) == len(xb) and len(xa) >= 3 else float("nan")
        tests.append([axis, metric, xa, xb, eff, p, None])
        fmt = "%.4f" if metric == "decode" else "%.1f"
        print(f"[stats] {axis} {metric}: {REF} n={len(xa)} [{' '.join(fmt % v for v in xa)}] | {ARM} n={len(xb)} [{' '.join(fmt % v for v in xb)}]")
# Benjamini-Hochberg over the declared family of 4 (tests not run yet count in the denominator)
m = len(AXES) * len(METRICS)
done = sorted((t for t in tests if t[5] == t[5]), key=lambda t: t[5])
qprev = 1.0
for rank in range(len(done), 0, -1):
    t = done[rank - 1]
    qprev = min(qprev, t[5] * m / rank)
    t[6] = qprev
for axis in AXES:
    ts = [t for t in tests if t[0] == axis]
    if not ts:
        continue
    for t in ts:
        stat_name = "mean" if axis == "multi" else "median"
        ra, rb = stat(axis, t[2]), stat(axis, t[3])
        q = "n/a" if t[6] is None else f"{t[6]:.4f}"
        print(f"[stats] {axis} {t[1]}: {stat_name} {REF} {ra:.4f} -> {ARM} {rb:.4f}, effect {t[4]:+.2f} %, p {t[5]:.4f}, q {q} (family of {m})")
    real = [t for t in ts if t[6] is not None and t[6] < Q and abs(t[4]) >= FLOOR]
    up = [t for t in real if t[4] > 0]; down = [t for t in real if t[4] < 0]
    verdict = "mixed" if up and down else "improves" if up else "REGRESSES" if down else "neutral"
    n_ok = all(len(t[2]) == len(t[3]) for t in ts)
    print(f"[stats] axis {axis}: {verdict}{'' if n_ok else ' (unequal n, provisional)'}")
