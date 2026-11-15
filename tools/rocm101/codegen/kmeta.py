#!/usr/bin/env python3
"""kmeta.py ELF... -- per kernel of an AMDGPU code object: stack bytes, spilled VGPRs/SGPRs, register counts, code bytes."""
import re, subprocess, sys
LL = '/opt/rocm/core-10.1/lib/llvm/bin/'
def meta(path):
    notes = subprocess.run([LL + 'llvm-readelf', '--notes', path], capture_output=True, text=True).stdout
    syms = subprocess.run([LL + 'llvm-readelf', '-s', '-W', path], capture_output=True, text=True).stdout
    size = {}
    for ln in syms.splitlines():
        f = ln.split()
        if len(f) >= 8 and f[3] == 'FUNC':
            size[f[7]] = int(f[2])
    out = {}
    cur = None
    for ln in notes.splitlines():
        m = re.match(r'\s+-?\s*\.(\w+):\s*(.*)', ln)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip().strip("'\"")
        if k == 'name':
            cur = out.setdefault(v, {})
        elif cur is not None and k in ('private_segment_fixed_size', 'vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count'):
            cur[k] = int(v)
    for n in out:
        out[n]['code'] = size.get(n, 0)
    return out
a, b = meta(sys.argv[1]), meta(sys.argv[2])
pat = sys.argv[3] if len(sys.argv) > 3 else ''
print("%-52s %s" % ("kernel", "stack B | vgpr spilled | sgpr spilled | vgprs | code B      (ROCm 10.0 -> 10.1)"))
for n in a:
    if pat and not re.search(pat, n):
        continue
    x, y = a[n], b.get(n)
    if y is None:
        print("%-52s only in 10.0" % n[:52]); continue
    short = re.sub(r'^_ZL\d+', '', n)[:52]
    f = lambda k: "%d -> %d" % (x.get(k, 0), y.get(k, 0))
    print("%-52s %-10s | %-10s | %-9s | %-10s | %s" % (short, f('private_segment_fixed_size'), f('vgpr_spill_count'), f('sgpr_spill_count'), f('vgpr_count'), f('code')))
