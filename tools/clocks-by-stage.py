#!/usr/bin/env python3
"""clocks-by-stage.py TAG -- per-stage summary of <TAG>-clocks.txt using the STAGE timestamps in <TAG>.progress."""
import sys, re
B = '/root/rocm-tests/bench'; tag = sys.argv[1]
prog = [l.split() for l in open(f'{B}/{tag}.progress') if l.strip()]
bounds = []; prev_t = prog[0][0][11:19]; prev_name = 'START'
for l in prog[1:]:
    t = l[0][11:19]; name = ' '.join(l[1:4]).rstrip(':') if l[1] == 'STAGE' else l[1]
    bounds.append((prev_name if prev_name != 'START' else name.replace('STAGE ', ''), prev_t, t)); prev_t, prev_name = t, None
# name each interval by the STAGE line that closes it
names = []
for l in prog[1:]:
    if l[1] == 'STAGE':
        toks = []
        for t in l[2:]:
            toks.append(t.rstrip(':'))
            if t.endswith(':'): break
        names.append(' '.join(toks))
    else: names.append(l[1])
samples = [l.split() for l in open(f'{B}/{tag}-clocks.txt') if l.strip()]
print(f'| stage | window (UTC) | samples | samples with a die at 1000 MHz | max junction C (0b/0e/1b/1e) | max W (0b/0e/1b/1e) | mean W of dies at 1730 MHz |')
print('|---|---|---:|---:|---|---|---:|')
for (name, t0, t1), nm in zip(bounds, names):
    win = [s for s in samples if t0 <= s[0] <= t1]
    if not win: continue
    low = sum(1 for s in win if any('1000Mhz' in f for f in s[1:5]))
    mt = [0]*4; mp = [0]*4; hi_p = []
    for s in win:
        for i, f in enumerate(s[1:5]):
            clk, temp, pw = f.split(':')[1].split('/'); t = int(temp.rstrip('C')); p = int(pw.rstrip('W'))
            mt[i] = max(mt[i], t); mp[i] = max(mp[i], p)
            if clk == '1730Mhz': hi_p.append(p)
    print(f'| {nm} | {t0}-{t1} | {len(win)} | {low} ({100*low/len(win):.0f}%) | {"/".join(map(str, mt))} | {"/".join(map(str, mp))} | {sum(hi_p)/len(hi_p):.0f} |')
