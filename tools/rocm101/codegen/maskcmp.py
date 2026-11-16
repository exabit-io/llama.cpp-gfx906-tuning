#!/usr/bin/env python3
"""maskcmp.py BUILD_A BUILD_B -- every gfx906 function whose bytes differ between two builds (fncmp.py's list): is the
difference more than a moved address? One full disassembly per object that has a differing function (TMPDIR on a real
disk); a function counts as "only immediates differ" when its instruction text is the same after masking numbers.
Why: two builds of the same sources from two source DIRECTORIES differ in about 1,600 functions with clang 24 (the
directory name is in the string pool, strings move, and every PC-relative reference to one changes: s_add_u32 only).
fncmp.py alone reports those as different. Used for the -r2 package against the -r1 build (2026-11-16)."""
import collections, glob, os, re, subprocess, sys, tempfile
src = open('/root/rocm101-migration-20261114/codegen/fncmp.py').read()
ns = {}; exec(src[:src.index('a, b = sys.argv[1], sys.argv[2]')], ns); funcs = ns['funcs']; T = ns['T']
OD = '/opt/rocm/core-10.1/lib/llvm/bin/llvm-objdump'
def dis(obj, tag):
    elf = os.path.join(os.environ['TMPDIR'], 'mask-%s.elf' % tag)
    subprocess.run([sys.executable, '-I', '/root/rocm101-migration-20261114/codegen/carve.py', obj, elf], capture_output=True)
    out = subprocess.run([OD, '-d', '--no-leading-addr', '--no-show-raw-insn', elf], capture_output=True, text=True).stdout
    f = {}; cur = None
    for l in out.splitlines():
        m = re.match(r'^[0-9a-f]* ?<(.+)>:$', l)
        if m: cur = m.group(1); f[cur] = []; continue
        if cur is None or not l.strip(): continue
        l = re.sub(r'//.*$', '', l); l = re.sub(r'<[^>]*>', '', l).strip()
        f[cur].append(l)
    return f
a, b = sys.argv[1], sys.argv[2]
res = collections.Counter(); real = collections.defaultdict(list); ops = collections.Counter()
for o in sorted(glob.glob(a + '/' + T + '/**/*.cu.o', recursive=True)):
    rel = os.path.relpath(o, a + '/' + T); ob = b + '/' + T + '/' + rel
    fa, fb = funcs(o), funcs(ob)
    names = [n for n in fa if n in fb and fa[n] != fb[n]]
    if not names: continue
    da, db = dis(o, 'a'), dis(ob, 'b')
    for n in names:
        x, y = da.get(n), db.get(n)
        if x is None or y is None: res['not disassembled'] += 1; real[rel].append(n); continue
        if x == y: res['text identical (data in the function differs)'] += 1; continue
        mx = [re.sub(r'0x[0-9a-f]+|\b\d+\b', 'N', l) for l in x]; my = [re.sub(r'0x[0-9a-f]+|\b\d+\b', 'N', l) for l in y]
        if mx == my:
            res['only immediates differ'] += 1
            for l, m in zip(x, y):
                if l != m: ops[l.split()[0]] += 1
        else: res['instructions differ'] += 1; real[rel].append(n)
print(dict(res)); print('opcodes of the lines whose immediates differ:', dict(ops.most_common(8)))
for rel, ns_ in real.items(): print('%s: %d functions with other instructions' % (rel, len(ns_)), sorted(set(re.sub(r'IL?[a-z0-9_]*.*', '', n) for n in ns_)))
