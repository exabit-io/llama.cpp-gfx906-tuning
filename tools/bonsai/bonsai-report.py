#!/usr/bin/env python3
# bonsai-report.py W TAG -- the table of bonsai-time.sh: per cell and arm the runs, the validity rule (V), peak VRAM,
# DC power in the decode phase, tokens per joule, evictions; and the Q8_0 reference of the acceptance session.
import collections, datetime as dt, glob, os, re, sys
w, tag = sys.argv[1:3]
runs = []
p = '%s/runs-%s.tsv' % (w, tag)
if os.path.exists(p):
    for l in open(p):
        f = l.rstrip('\n').split('\t')
        if len(f) >= 9: runs.append(dict(S=int(f[0]), D=int(f[1]), ntg=int(f[2]), arm=f[3], blk=int(f[4]), t0=int(f[5]), t1=int(f[6]), dec=float(f[7]), pre=float(f[8])))
# SMC DC power: "HH:MM:SS PZ0G=..." with a dated header line
smc = []; day = None; last = None
p = '%s/time-%s-smc.log' % (w, tag)
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
p = '%s/time-%s-vram.log' % (w, tag)
if os.path.exists(p):
    for l in open(p):
        f = l.split()
        try: vram.append((int(f[0]), [int(x) / 2**30 for x in f[1:5]]))
        except Exception: pass
ev = collections.defaultdict(list)
p = '%s/ev-%s.log' % (w, tag)
if os.path.exists(p):
    for l in open(p):
        if l.startswith('#'): continue
        f = l.split()
        if len(f) == 5 and f[3].startswith('llama-batched'): ev[f[2]].append((int(f[0]), int(f[4])))
procs = [(v[0][0], v[-1][0], v[-1][1], v) for v in ev.values()]
cells = collections.OrderedDict()
for r in runs: cells.setdefault((r['S'], r['D']), collections.OrderedDict()).setdefault(r['arm'], []).append(r)
print('[report] per run: cell arm block | decode tok/s per slot | prefill tok/s | seconds | peak VRAM per die GiB | DC W in decode | evicted ms (seconds with >= 100 ms)')
summary = []
for (S, D), arms in cells.items():
    for arm, rs in arms.items():
        for r in rs:
            tdec = r['ntg'] / r['dec']                       # seconds of the decode phase
            a, b = r['t1'] - 2 - tdec, r['t1'] - 2
            pw = [v for (t, v) in smc if a <= t <= b]
            r['dcw'] = sum(pw) / len(pw) if pw else float('nan')
            vv = [x for (t, x) in vram if r['t0'] <= t <= r['t1']]
            r['vram'] = [max(x[i] for x in vv) for i in range(4)] if vv else [float('nan')] * 4
            mine = [x for x in procs if x[0] >= r['t0'] - 2 and x[1] <= r['t1'] + 2]
            r['ev'] = sum(x[2] for x in mine) if mine else -1
            steps = []
            for x in mine:
                v = x[3]
                for i in range(1, len(v)):
                    if v[i][1] - v[i-1][1] >= 100: steps.append('+%ds:%d' % (v[i][0] - v[0][0], v[i][1] - v[i-1][1]))
            print('[report]   %d x %d %-4s b%d | %9.4f | %8.1f | %4d | %s | %6.1f | %d (%s)' % (S, D, arm, r['blk'], r['dec'], r['pre'], r['t1'] - r['t0'],
                  ' '.join('%.1f' % x for x in r['vram']), r['dcw'], r['ev'], ' '.join(steps)))
        dec = [r['dec'] for r in rs]; pre = [r['pre'] for r in rs]
        md, mp = sum(dec) / len(dec), sum(pre) / len(pre)
        sd, sp = (max(dec) - min(dec)) / md * 100, (max(pre) - min(pre)) / mp * 100
        ok = len(rs) >= 3 and sp <= 0.5 and sd <= 1.0
        dcw = sum(r['dcw'] for r in rs) / len(rs)
        vr = max(max(r['vram']) for r in rs)
        summary.append((S, D, arm, len(rs), md, sd, mp, sp, ok, dcw, vr))
print()
print('[report] summary: cell arm n | decode per slot (spread) | aggregate decode | prefill (spread) | (V) | peak VRAM on the fullest die | DC W in decode | decode tokens per joule | R3.1 (12 tok/s per request)')
for S, D, arm, n, md, sd, mp, sp, ok, dcw, vr in summary:
    print('[report]   %d x %-6d %-4s n=%d | %8.3f (%.2f %%) | %8.2f | %8.1f (%.2f %%) | %s | %5.1f GiB | %6.1f W | %.3f | %s' % (
        S, D, arm, n, md, sd, S * md, mp, sp, 'readable' if ok else 'NOT READABLE', vr, dcw, S * md / dcw if dcw == dcw else float('nan'),
        'over' if md >= 12 else 'UNDER'))
# the Q8_0 reference of the acceptance session (same package, same boot), arm pkg
print()
print('[report] Qwen3.8-27B Q8_0 on the same package, four dies, tensor split: acceptance session 2026-11-27/28 (other session, same boot)')
ref = {}
for name, files in (('acceptance', glob.glob('/root/rebase-v110/screen-acc-s*.tsv')), ('clean rerun dp2', glob.glob('/root/rebase-v110/screen-dp2-s*.tsv'))):
    d = collections.defaultdict(list)
    for path in files:
        for l in open(path):
            f = l.rstrip('\n').split('\t')
            if len(f) >= 5 and f[1] == 'pkg':
                try: d[f[0]].append((float(f[3]), float(f[4])))
                except ValueError: pass
    for cell, v in sorted(d.items()):
        md = sum(x[0] for x in v) / len(v); mp = sum(x[1] for x in v) / len(v)
        sp = (max(x[1] for x in v) - min(x[1] for x in v)) / mp * 100
        print('[report]   %-12s %-16s n=%d | decode per slot %8.3f | prefill %8.1f (spread %.2f %%)' % (cell, name, len(v), md, mp, sp))
        ref.setdefault(cell, {})[name] = (md, mp, sp)
print()
print('[report] Bonsai Q2_0 against Qwen3.8-27B Q8_0 (four dies, tensor split) at the same cell (T2: said only beyond 3 %):')
for S, D, arm, n, md, sd, mp, sp, ok, dcw, vr in summary:
    cell = '%d x %d' % (S, D)
    if arm == 'q8t' or cell not in ref: continue
    r = ref[cell].get('acceptance')
    c = ref[cell].get('clean rerun dp2')
    pre_ref = (c or r)[1] if (r and r[2] > 0.5 and c) else r[1]
    dd, dp = (md / r[0] - 1) * 100, (mp / pre_ref - 1) * 100
    word = lambda x: 'faster' if x > 3 else ('slower' if x < -3 else 'level within what two sessions can show')
    print('[report]   %-12s %-4s decode per slot %.3f against %.3f: %+.1f %% (%s); prefill %.1f against %.1f: %+.1f %% (%s)' % (cell, arm, md, r[0], dd, word(dd), mp, pre_ref, dp, word(dp)))
