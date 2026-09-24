#!/usr/bin/env python3
"""sweep-report.py — 3-way sanity sweep from sweep.tsv: master vs the pure substrate (merge-v0.5.0) vs stock v0.5.0.
n=1 per build per cell: a sanity check, not a bin. Flags: a cell that fails on some builds only, and master more than 5%
slower than either yardstick on decode or prefill (2.5x the box's +/-2% run-to-run band, so n=1 noise alone should not
trip it)."""
import sys, collections
rows = collections.defaultdict(dict); order = []
for line in open(sys.argv[1] if len(sys.argv) > 1 else '/root/night-20260919/sweep.tsv'):
    f = line.rstrip('\n').split('\t')
    if len(f) < 6:
        continue
    key = (f[0], f[1])
    if key not in rows:
        order.append(key)
    rows[key][f[2]] = f[3:6]
def num(r, i):
    try: return float(r[i])
    except (TypeError, ValueError, IndexError): return None
def pct(a, b): return f"{100*(a/b-1):+6.1f}%" if a and b else '     -'
B = ('stock', 'substrate', 'master')
for metric, idx in (('decode tok/s per sequence', 1), ('prefill t/s', 2)):
    print(f"\n{metric}")
    print(f"{'model':30} {'cell':8} {'stock':>9} {'substrate':>9} {'master':>9} {'m vs sub':>9} {'m vs stock':>10}  flag")
    for key in order:
        r = rows[key]
        v = {b: num(r.get(b), idx) for b in B}
        st = {b: (r.get(b) or ['missing'])[0] for b in B}
        flag = ''
        if any(s != 'ok' for s in st.values()):
            flag = ' '.join(f"{b}={st[b]}" for b in B if st[b] != 'ok')
        elif v['master'] and ((v['substrate'] and v['master']/v['substrate'] < 0.95) or (v['stock'] and v['master']/v['stock'] < 0.95)):
            flag = 'REGRESSION >5%'
        f2 = lambda x: f"{x:9.2f}" if x else '        -'
        print(f"{key[0]:30} {key[1]:8} {f2(v['stock'])} {f2(v['substrate'])} {f2(v['master'])} {pct(v['master'], v['substrate']):>9} {pct(v['master'], v['stock']):>10}  {flag}")
