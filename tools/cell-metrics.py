#!/usr/bin/env python3
"""Extract decode and prefill from a llama-batched-bench table at FULL PRECISION.

WHY THIS EXISTS (2026-09-21). Every survey script until now read the rate columns S_PP and S_TG,
which the tool prints to TWO DECIMALS. On the single-user axis that rounding produced TIED values
(40.19 three times in one arm), and ties make the permutation null degenerate for a median statistic:
a 3-1 split has the SAME median as the perfect 4-0 split, so ten of 70 arrangements matched the
observed statistic instead of two. A +25.9% effect on completely non-overlapping data came out at
p=0.1429 and "failed" its pre-registered test.

The timing columns T_PP and T_TG carry three decimals, so deriving the rates removes the ties:

    decode tok/s/slot = n_tg / T_TG          (per sequence; the table's S_TG is the aggregate)
    prefill t/s       = n_pp * batch / T_PP

I nearly "fixed" this by swapping in a rank test and an endpoint hierarchy. Both would have worked,
and both would have been post-hoc changes to the analysis after seeing the result. The real defect was
measurement precision, and correcting it rescued every verdict under the ORIGINAL pre-registered rule.
Prefer fixing the instrument over weakening the test.

usage: cell-metrics.py FILE.md DEPTH [SLOTS]   -> prints "decode<TAB>prefill" or exits 1
"""
import re, sys
if len(sys.argv) < 3:
    sys.exit(__doc__)
path, depth = sys.argv[1], int(sys.argv[2])
slots = int(sys.argv[3]) if len(sys.argv) > 3 else None
row = None
for line in open(path):
    if re.match(r'^\|\s*%d\s' % depth, line):
        row = [c.strip() for c in line.split('|')]
if row is None:
    print("no result row for depth %d in %s" % (depth, path), file=sys.stderr); sys.exit(1)
try:
    n_pp, n_tg, b = int(row[1]), int(row[2]), int(row[3])
    t_pp, t_tg = float(row[5]), float(row[7])
except (ValueError, IndexError) as e:
    print("unparseable row in %s: %s" % (path, e), file=sys.stderr); sys.exit(1)
if slots is not None and b != slots:
    print("batch in table (%d) != slots requested (%d) in %s" % (b, slots, path), file=sys.stderr); sys.exit(1)
if t_pp <= 0 or t_tg <= 0:
    print("non-positive timing in %s" % path, file=sys.stderr); sys.exit(1)
print("%.6f\t%.4f" % (n_tg / t_tg, n_pp * b / t_pp))
