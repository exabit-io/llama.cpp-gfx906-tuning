#!/usr/bin/env python3
"""sub-stats.py TAG -- reads the substrate comparison's result files (screen-TAG-*.tsv of screen.sh, TAG-mtp.tsv of
substrate-v110.sh) and prints, per pair of arms, every cell and metric: n, the two means, the effect (ratio of the
geometric means - 1) with its 95 % confidence interval, the p value, the Holm-adjusted p value and the verdict.

Method (fixed in substrate-v110.sh's header before the first run): every block runs every arm once in a fresh random
order, so the samples of two arms are PAIRED by block; the statistic is the mean of ln(a_i / b_i) over the blocks
where both arms have a value, tested against 0 with Student's t (df = n - 1). Holm's correction is applied over all
cell x metric rows of one pair. Words:
  MAJOR PENALTY / major gain   adjusted p < 0.05 and the whole interval beyond -LINE / +LINE (default 2 %)
  penalty / gain               adjusted p < 0.05, the interval excludes 0 but not the line
  no difference shown          adjusted p >= 0.05
A row with fewer than 3 pairs is printed without a verdict.
"""
import collections
import glob
import math
import os
import sys

from scipy import stats

W = os.environ.get("SUBW", "/root/rebase-v110")
tag = sys.argv[1]
LINE = float(os.environ.get("LINE", "2.0"))
PAIRS = [p.split(":") for p in os.environ.get(
    "PAIRS", "merge:mxxm master:mxxm master:merge stock:mxxm stock:merge master:stock").split()]

# data[(cell, metric)][arm][block] = value
data = collections.defaultdict(lambda: collections.defaultdict(dict))
for path in sorted(glob.glob("%s/screen-%s-*.tsv" % (W, tag))):
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 5:
            continue
        try:
            dec, pre = float(f[3]), float(f[4])
        except ValueError:
            continue
        data[(f[0], "decode tok/s per slot")][f[1]][f[2]] = dec
        data[(f[0], "prefill tok/s")][f[1]][f[2]] = pre
acc = collections.defaultdict(dict)
mtp = "%s/%s-mtp.tsv" % (W, tag)
if os.path.exists(mtp):
    for line in open(mtp):
        f = line.rstrip("\n").split("\t")
        if len(f) < 6 or f[0] != "mtp":
            continue
        try:
            pre, dec = float(f[3]), float(f[4])
        except ValueError:
            continue
        data[("MTP 1 stream", "prefill tok/s")][f[1]][f[2]] = pre
        data[("MTP 1 stream", "decode tok/s (mean of the requests)")][f[1]][f[2]] = dec
        acc[f[1]][f[2]] = f[5]

arms = sorted({a for d in data.values() for a in d})
print("arms with data: %s" % " ".join(arms))
print()
print("Per arm: n, mean (min-max)")
for key in data:
    print("  %s, %s" % key)
    for a in arms:
        v = list(data[key].get(a, {}).values())
        if v:
            print("    %-7s n=%d  %10.4f  (%.4f-%.4f, span %.2f %%)" % (a, len(v), sum(v) / len(v), min(v), max(v), 100 * (max(v) / min(v) - 1)))
if acc:
    print("  MTP accepted/drafted tokens per request (first block of each arm; the texts of two trees differ):")
    for a in arms:
        if a in acc:
            print("    %-7s %s" % (a, acc[a][sorted(acc[a])[0]]))
print()

for a, b in PAIRS:
    rows = []
    for key in data:
        da, db = data[key].get(a, {}), data[key].get(b, {})
        blocks = sorted(set(da) & set(db))
        if not blocks:
            continue
        r = [math.log(da[k] / db[k]) for k in blocks]
        n = len(r)
        m = sum(r) / n
        row = {"key": key, "n": n, "ma": sum(da[k] for k in blocks) / n, "mb": sum(db[k] for k in blocks) / n, "eff": 100 * (math.exp(m) - 1),
               "sep": "above" if min(da.values()) > max(db.values()) else "below" if max(da.values()) < min(db.values()) else "overlap"}
        if n >= 3:
            sd = math.sqrt(sum((x - m) ** 2 for x in r) / (n - 1))
            se = sd / math.sqrt(n)
            if se == 0:
                row.update(lo=row["eff"], hi=row["eff"], p=0.0 if m != 0 else 1.0)
            else:
                tc = stats.t.ppf(0.975, n - 1)
                row.update(lo=100 * (math.exp(m - tc * se) - 1), hi=100 * (math.exp(m + tc * se) - 1), p=2 * stats.t.sf(abs(m / se), n - 1))
        rows.append(row)
    if not rows:
        continue
    tested = sorted((r for r in rows if "p" in r), key=lambda r: r["p"])
    prev = 0.0
    for i, r in enumerate(tested):
        prev = max(prev, min(1.0, (len(tested) - i) * r["p"]))
        r["padj"] = prev
    print("%s against %s (effect = %s / %s - 1; Holm over %d rows; line %.1f %%)" % (a, b, a, b, len(tested), LINE))
    worst = None
    for r in rows:
        head = "  %-13s %-36s n=%d  %10.4f vs %10.4f  %+7.2f %%" % (r["key"][0], r["key"][1], r["n"], r["ma"], r["mb"], r["eff"])
        if "p" not in r:
            print(head + "  (fewer than 3 pairs: no test)")
            continue
        if r["padj"] >= 0.05:
            word = "no difference shown"
        elif r["hi"] < -LINE:
            word = "MAJOR PENALTY"
        elif r["lo"] > LINE:
            word = "major gain"
        elif r["hi"] < 0:
            word = "penalty"
        elif r["lo"] > 0:
            word = "gain"
        else:
            word = "no difference shown"
        if word == "MAJOR PENALTY" and (worst is None or r["eff"] < worst["eff"]):
            worst = r
        print(head + "  [%+.2f, %+.2f]  p=%.2g adj=%.2g  samples %s  %s" % (r["lo"], r["hi"], r["p"], r["padj"], r["sep"], word))
    print("  => %s" % ("MAJOR PENALTY in at least one row (largest: %s, %s, %+.2f %%)" % (worst["key"][0], worst["key"][1], worst["eff"])
                       if worst else "no major penalty of %s against %s in any row tested" % (a, b)))
    print()
