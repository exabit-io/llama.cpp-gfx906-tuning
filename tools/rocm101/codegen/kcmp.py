#!/usr/bin/env python3
"""kcmp.py NAME=OBJECT ... -- the head-256 tile kernels the service runs (<1,2> decode, <4,2> MTP verification, <16,2>
prefill) of several builds' tile objects: which builds have the same instructions (numbers masked, as maskcmp.py does)."""
import hashlib, os, re, subprocess, sys
W = '/root/rocm101-migration-20261114'; OD = '/opt/rocm/core-10.1/lib/llvm/bin/llvm-objdump'
def dis(obj, tag):
    elf = os.path.join(os.environ['TMPDIR'], 'kcmp-%s.elf' % tag)
    subprocess.run([sys.executable, '-I', W + '/codegen/carve.py', obj, elf], capture_output=True)
    out = subprocess.run([OD, '-d', '--no-leading-addr', '--no-show-raw-insn', elf], capture_output=True, text=True).stdout
    os.unlink(elf)
    f = {}; cur = None
    for l in out.splitlines():
        m = re.match(r'^[0-9a-f]* ?<(.+)>:$', l)
        if m: cur = m.group(1); f[cur] = []; continue
        if cur is None or not l.strip(): continue
        l = re.sub(r'//.*$', '', l); l = re.sub(r'<[^>]*>', '', l).strip()
        f[cur].append(l)
    return f
arms = [a.split('=', 1) for a in sys.argv[1:]]
d = {n: dis(o, n) for n, o in arms}
for nc1 in (1, 2, 4, 8, 16):
    key = 'flash_attn_tileILi256ELi256ELi%dELi2ELb0E' % nc1
    row = []; seen = {}
    for n, _ in arms:
        fs = [k for k in d[n] if key in k]
        if len(fs) != 1: row.append('%s:?(%d)' % (n, len(fs))); continue
        exact = hashlib.sha1('\n'.join(d[n][fs[0]]).encode()).hexdigest()[:6]
        masked = hashlib.sha1('\n'.join(re.sub(r'\b(0x[0-9a-f]+|\d+)\b', 'N', l) for l in d[n][fs[0]]).encode()).hexdigest()[:6]
        g = seen.setdefault(masked, chr(ord('A') + len(seen)))
        row.append('%s:%s(%d insn, exact %s)' % (n, g, len(d[n][fs[0]]), exact))
    print('<%d,2>  %s' % (nc1, '  '.join(row)))
