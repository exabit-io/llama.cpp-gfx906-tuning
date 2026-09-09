#!/usr/bin/env python3
"""compare-27b-runs.py OLD_TAG NEW_TAG  -- side-by-side of the Qwen 27B benchmark files in /root/rocm-tests/bench.
Reads <tag>.md (llama-bench), <tag>-batched.md (llama-batched-bench), <tag>-server-{tp4,layer4,tp2}.md (server-bench.py)."""
import re, sys, os
B = '/root/rocm-tests/bench'
def rows(path):
    try: lines = open(path, errors='ignore').read().splitlines()
    except OSError: return []
    out = []
    for l in lines:
        if not l.startswith('|'): continue
        cells = [c.strip() for c in l.strip().strip('|').split('|')]
        if not cells or set(cells[0]) <= set('-:') or not any(ch.isdigit() for ch in ''.join(cells)): continue
        out.append(cells)
    return out
def num(s):
    m = re.match(r'\s*([-+]?\d+(?:\.\d+)?)', s); return float(m.group(1)) if m else None
def llama_bench(tag):
    d = {}
    for c in rows(f'{B}/{tag}.md'):
        if len(c) >= 11 and c[3] == 'ROCm': d[(c[8], c[9])] = num(c[10])
    return d
def batched(tag):
    d = {}
    for c in rows(f'{B}/{tag}-batched.md'):
        if len(c) >= 10 and c[0].isdigit(): d[(int(c[0]), int(c[2]))] = (num(c[5]), num(c[7]), num(c[9]))
    return d
def server(tag, mode):
    d = {}
    for c in rows(f'{B}/{tag}-server-{mode}.md'):
        if len(c) >= 9 and c[0].isdigit(): d[int(c[0])] = (num(c[3]), num(c[4]), num(c[6]), num(c[7]))   # agg gen, total tok/s, per-req gen, TTFT
    return d
def pct(a, b):
    return '' if a in (None, 0) or b is None else f'{(b - a) / a * 100:+.1f}%'
def fmt(v): return '(missing)' if v is None else (f'{v:.2f}' if v < 100 else f'{v:.1f}')
old, new = sys.argv[1], sys.argv[2]
print(f'OLD = {old}   NEW = {new}\n')
print(f'## llama-bench (t/s)\n| dev | test | old | new | change |\n|---|---|---:|---:|---:|')
lo, ln = llama_bench(old), llama_bench(new)
for k in lo: print(f'| {k[0]} | {k[1]} | {fmt(lo[k])} | {fmt(ln.get(k))} | {pct(lo[k], ln.get(k))} |')
print(f'\n## llama-batched-bench, 4 dies tensor split (S_PP / S_TG / S t/s)\n| PP | B | old pp | new pp | change | old tg | new tg | change | old total | new total | change |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
bo, bn = batched(old), batched(new)
for k in bo:
    o = bo[k]; n = bn.get(k, (None, None, None))
    print(f'| {k[0]} | {k[1]} | {fmt(o[0])} | {fmt(n[0])} | {pct(o[0], n[0])} | {fmt(o[1])} | {fmt(n[1])} | {pct(o[1], n[1])} | {fmt(o[2])} | {fmt(n[2])} | {pct(o[2], n[2])} |')
for mode in ('tp4', 'layer4', 'tp2'):
    so, sn = server(old, mode), server(new, mode)
    if not so and not sn: continue
    print(f'\n## server sweep {mode} (agg gen t/s | total tok/s | per-request gen t/s | TTFT s)\n| conc | old agg gen | new agg gen | change | old total | new total | change | old per-req | new per-req | change | old TTFT | new TTFT | change |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for c in sorted(set(so) | set(sn)):
        o = so.get(c, (None,)*4); n = sn.get(c, (None,)*4)
        print(f'| {c} | {fmt(o[0])} | {fmt(n[0])} | {pct(o[0], n[0])} | {fmt(o[1])} | {fmt(n[1])} | {pct(o[1], n[1])} | {fmt(o[2])} | {fmt(n[2])} | {pct(o[2], n[2])} | {fmt(o[3])} | {fmt(n[3])} | {pct(o[3], n[3]) and pct(o[3], n[3])} |')
