#!/usr/bin/env python3
# perdie-report.py W TAG -- the table of bonsai-perdie.sh (rules V, R1-R4 of its header).
import collections, datetime as dt, glob, os, re, sys
w, tag = sys.argv[1:3]
runs = []
p = '%s/perdie-%s.tsv' % (w, tag)
if os.path.exists(p):
    for l in open(p):
        f = l.rstrip('\n').split('\t')
        if len(f) >= 10:
            runs.append(dict(mode=f[0], S=int(f[1]), D=int(f[2]), ntg=int(f[3]), die=int(f[4]), blk=int(f[5]), t0=int(f[6]), t1=int(f[7]), dec=float(f[8]), pre=float(f[9])))
smc = []; day = None; last = None
p = '%s/perdie-%s-smc.log' % (w, tag)
if os.path.exists(p):
    for l in open(p):
        if l.startswith('#'):
            m = re.match(r'# (\S+) smc-log start', l)
            if m: day = dt.datetime.fromisoformat(m.group(1)).date(); last = None
            continue
        f = l.split()
        try:
            t = dt.datetime.combine(day, dt.time.fromisoformat(f[0]), tzinfo=dt.timezone.utc)
            kv = dict(x.split('=') for x in f[1:] if '=' in x); v = float(kv['PZ0G'])
        except Exception: continue
        if last is not None and t < last - dt.timedelta(hours=12): day += dt.timedelta(days=1); t += dt.timedelta(days=1)
        last = t; smc.append((t.timestamp(), v))
vram = []
p = '%s/perdie-%s-vram.log' % (w, tag)
if os.path.exists(p):
    for l in open(p):
        f = l.split()
        try: vram.append((int(f[0]), [int(x) / 2**30 for x in f[1:5]]))
        except Exception: pass
steps = collections.OrderedDict()
for r in runs: steps.setdefault((r['mode'], r['S']), []).append(r)
mean = lambda v: sum(v) / len(v)
res = {}
print('[report] per step: per die the runs (decode tok/s per slot / prefill tok/s), spread between the runs, rule (V)')
for (mode, S), rs in steps.items():
    dies = sorted({r['die'] for r in rs}); ok = True; rows = []
    for d in dies:
        v = sorted([r for r in rs if r['die'] == d], key=lambda r: r['blk'])
        dec = [r['dec'] for r in v]; pre = [r['pre'] for r in v]
        sd = (max(dec) - min(dec)) / mean(dec) * 100; sp = (max(pre) - min(pre)) / mean(pre) * 100
        good = len(v) >= 2 and sd <= 1.0 and sp <= 0.5; ok = ok and good
        rows.append((d, mean(dec), mean(pre), sd, sp))
        print('[report]   %s:%d die %d: %s | spread decode %.2f %%, prefill %.2f %% %s' % (mode, S, d, '  '.join('%.3f/%.1f' % (r['dec'], r['pre']) for r in v), sd, sp, '' if good else '<- outside (V)'))
    # decode phase power: the last ntg/dec seconds of each run (minus 2 s of teardown), mean over runs
    pws = []; vr = [0.0] * 4
    for blk in sorted({r['blk'] for r in rs}):
        v = [r for r in rs if r['blk'] == blk]; t1 = v[0]['t1']; t0 = v[0]['t0']
        tdec = max(r['ntg'] / r['dec'] for r in v)
        pw = [x for (t, x) in smc if t1 - 2 - tdec <= t <= t1 - 2]
        if pw: pws.append(mean(pw))
        for (t, x) in vram:
            if t0 <= t <= t1: vr = [max(a, b) for a, b in zip(vr, x)]
    aggd = sum(S * x[1] for x in rows); aggp = sum(x[2] for x in rows); slot = mean([x[1] for x in rows])
    dcw = mean(pws) if pws else float('nan')
    res[(mode, S)] = dict(slot=slot, aggd=aggd, aggp=aggp, ok=ok, dcw=dcw, vr=vr, rows=rows, n=len({r['blk'] for r in rs}))
print()
print('[report] summary: step | total slots x depth | decode per slot (mean of the dies) | aggregate decode | aggregate prefill | (V) | peak VRAM per die GiB | DC W in decode | tokens per joule | R3.1')
for (mode, S), x in res.items():
    nd = len(x['rows'])
    print('[report]   %s:%d | %d x 65536 | %7.3f | %8.2f | %8.1f | %s | %s | %6.1f | %.3f | %s' % (mode, S, nd * S, x['slot'], x['aggd'], x['aggp'],
          'readable' if x['ok'] else 'NOT READABLE', ' '.join('%.1f' % v for v in x['vr'][:nd] if v > 0) or '-', x['dcw'], x['aggd'] / x['dcw'] if x['dcw'] == x['dcw'] else float('nan'),
          'over 12' if min(r[1] for r in x['rows']) >= 12 else 'UNDER 12'))
print()
print('[report] Qwen3.8-27B Q8_0, four dies, tensor split, same package (acceptance session 2026-11-27/28: other session, same boot):')
ref = {}
for name, files in (('acceptance', glob.glob('/root/rebase-v110/screen-acc-s*.tsv')), ('clean partial rerun', glob.glob('/root/rebase-v110/screen-dp2-s*.tsv'))):
    d = collections.defaultdict(list)
    for path in files:
        for l in open(path):
            f = l.rstrip('\n').split('\t')
            if len(f) >= 5 and f[1] == 'pkg':
                try: d[f[0]].append((float(f[3]), float(f[4])))
                except ValueError: pass
    for cell, v in sorted(d.items()):
        if '65536' not in cell: continue
        md = mean([x[0] for x in v]); mp = mean([x[1] for x in v]); sp = (max(x[1] for x in v) - min(x[1] for x in v)) / mp * 100
        print('[report]   %-10s %-20s n=%d | decode per slot %7.3f | aggregate decode %7.2f | prefill %7.1f (spread %.2f %%)' % (cell, name, len(v), md, md * int(cell.split()[0]), mp, sp))
        ref.setdefault(cell, {})[name] = (md, mp, sp)
print()
word = lambda x: 'faster' if x > 3 else ('slower' if x < -3 else 'level within what two sessions can show')
for (mode, S), x in res.items():
    if mode != 'quad': continue
    cell = '%d x 65536' % (4 * S)
    if cell not in ref: print('[report] (R3) quad:%d = %s: no Q8_0 measurement at that cell' % (S, cell)); continue
    a = ref[cell].get('acceptance'); c = ref[cell].get('clean partial rerun')
    pre = c[1] if (a and a[2] > 0.5 and c) else a[1]
    dd = (x['slot'] / a[0] - 1) * 100; dp = (x['aggp'] / pre - 1) * 100
    print('[report] (R3) quad:%d = %s against Q8_0 tensor split: decode per slot %.3f against %.3f, %+.1f %% (%s); aggregate prefill %.1f against %.1f, %+.1f %% (%s)' % (S, cell, x['slot'], a[0], dd, word(dd), x['aggp'], pre, dp, word(dp)))
if ('quad', 1) in res and ('solo', 1) in res:
    q = [r for r in res[('quad', 1)]['rows'] if r[0] == 0][0]; s = res[('solo', 1)]['rows'][0]
    dd = (q[1] / s[1] - 1) * 100; dp = (q[2] / s[2] - 1) * 100
    print('[report] (R4) die 0 with three busy neighbours against alone: decode %+.2f %%, prefill %+.2f %% (a cost is said only beyond 2 %%)' % (dd, dp))
