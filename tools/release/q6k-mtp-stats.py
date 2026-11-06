#!/usr/bin/env python3
"""q6k-mtp-stats.py DIR -- phase M of q6k-campaign.sh. Written BEFORE the data (2026-10-22).

Unit = one server run (n=4 per arm). Per run, from the client's JSON:
  decode   wave 2 (all slots decoding a cached prompt together): mean over requests of generated / generation time, tok/s
  agg      wave 2: sum over requests of the same per-request rates, tok/s
  ttft     wave 1 (prompt prefilled): mean prompt time, s
  accept   wave 2 accepted / drafted (MTP arms only)
Run-level statistic: mean (multi-user), median (single-user, heavy-tailed stalls), as binstats.py.
Contrasts per profile: q6 vs q8 within each mode, mtp3 vs off within each quant; metrics decode, agg, ttft.
Test: two-sided exact permutation (4 vs 4 = 70 arrangements, p floor 0.0286); BH FDR over the whole family; REPORT ONLY.
"""
import glob, itertools, json, os, re, statistics as st, sys
d = sys.argv[1]
runs = {}
for f in sorted(glob.glob(os.path.join(d, 'q6k-m-*.json'))):
    m = re.match(r'q6k-m-(multi|single)-(q8|q6)-(off|mtp3)-(\d+)\.json', os.path.basename(f))
    if not m: continue
    prof, arm, mode, blk = m.groups()
    w = {x['wave']: x for x in json.load(open(f))}
    if 1 not in w or 2 not in w: continue
    r2 = w[2]['res']; r1 = w[1]['res']
    rates = [x['gn'] / (x['gms'] / 1000) for x in r2]
    acc = w[2]['acc']; am = re.match(r'(\d+) / (\d+)', acc or '')
    runs.setdefault((prof, arm, mode), []).append(dict(decode=st.mean(rates), agg=sum(rates),
        ttft=st.mean(x['pms'] / 1000 for x in r1), accept=(int(am[1]) / int(am[2])) if am and int(am[2]) else None))
def stat(prof, xs): return st.mean(xs) if prof == 'multi' else st.median(xs)
def perm_p(a, b, prof):
    obs = abs(stat(prof, a) - stat(prof, b)); pool = a + b; n = len(a); hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        x = [pool[i] for i in idx]; y = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1; hits += abs(stat(prof, x) - stat(prof, y)) >= obs - 1e-12
    return hits / tot
print("arm summaries (statistic over runs; min..max)")
print(f"{'profile':7} {'quant':5} {'mode':5} {'n':>2} {'decode tok/s/req':>22} {'agg tok/s':>22} {'TTFT s':>18} {'accepted/drafted':>17}")
for (prof, arm, mode), rs in sorted(runs.items()):
    def fmt(k, p=2):
        v = [r[k] for r in rs]; return f"{stat(prof, v):.{p}f} ({min(v):.{p}f}..{max(v):.{p}f})"
    acc = [r['accept'] for r in rs if r['accept'] is not None]
    print(f"{prof:7} {arm:5} {mode:5} {len(rs):>2} {fmt('decode'):>22} {fmt('agg', 1):>22} {fmt('ttft', 1):>18} {(f'{st.mean(acc):.3f}' if acc else '-'):>17}")
tests = []
for prof in ('multi', 'single'):
    for (a, b, label) in [(('q6', 'off'), ('q8', 'off'), 'Q6_K vs Q8_0, MTP off'), (('q6', 'mtp3'), ('q8', 'mtp3'), 'Q6_K vs Q8_0, MTP on'),
                          (('q8', 'mtp3'), ('q8', 'off'), 'MTP on vs off, Q8_0'), (('q6', 'mtp3'), ('q6', 'off'), 'MTP on vs off, Q6_K')]:
        A = runs.get((prof,) + a); Bv = runs.get((prof,) + b)
        if not A or not Bv: continue
        for k in ('decode', 'agg', 'ttft'):
            x = [r[k] for r in A]; y = [r[k] for r in Bv]
            ref = stat(prof, y); arm = stat(prof, x)
            tests.append([prof, label, k, len(y), len(x), ref, arm, (arm / ref - 1) * 100, perm_p(x, y, prof)])
m = len(tests); order = sorted(range(m), key=lambda i: tests[i][8]); q = [0.0] * m; prev = 1.0
for rank, i in reversed(list(enumerate(order, 1))):
    prev = min(prev, tests[i][8] * m / rank); q[i] = prev
print("\ncontrasts (ref -> arm, effect, exact permutation p, BH q over the family of", m, "tests)")
print(f"{'profile':7} {'contrast':24} {'metric':6} {'n':>5} {'ref':>9} {'arm':>9} {'effect':>8} {'p':>7} {'q':>7}")
for i, t in enumerate(tests):
    print(f"{t[0]:7} {t[1]:24} {t[2]:6} {t[3]}v{t[4]:<3} {t[5]:9.2f} {t[6]:9.2f} {t[7]:+7.2f}% {t[8]:7.4f} {q[i]:7.4f}")
