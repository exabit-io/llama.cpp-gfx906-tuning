#!/usr/bin/env python3
"""loopscan.py BUILD_A BUILD_B [MIN] -- every gfx906 device function of the two builds' HIP objects: scratch (spill)
traffic INSIDE LOOPS. spilldiff.py counts spilled registers per kernel, which does not say whether a reload is executed
once or per iteration; this counts the buffer_load_* (reload) and buffer_store_* (spill) instructions that lie inside a
backward branch's range. Prints the functions whose in-loop reloads differ by MIN (default 1) or more, by the change,
and a tally per object. Uses the ROCm 10.1 llvm-objdump for both builds. JOBS (env, default 8) objects at a time."""
import glob, multiprocessing, os, re, struct, subprocess, sys, tempfile
LL = '/opt/rocm/core-10.1/lib/llvm/bin/'
T = 'ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda'
def device_elf(path):
    b = open(path, 'rb').read(); i = 0
    while True:
        i = b.find(b'\x7fELF', i + 1)
        if i < 0: return None
        if b[i + 4] == 2 and struct.unpack_from('<H', b, i + 18)[0] == 224:
            shoff, = struct.unpack_from('<Q', b, i + 40); es, n = struct.unpack_from('<HH', b, i + 58)
            end = shoff + es * n
            for k in range(n):
                off, size = struct.unpack_from('<QQ', b, i + shoff + k * es + 24)
                if struct.unpack_from('<I', b, i + shoff + k * es + 4)[0] != 8: end = max(end, off + size)
            return b[i:i + end]
def scan(arg):
    """one object -> {function: (instructions, loops, reloads in loops, spills in loops, reloads, largest loop)}"""
    build, o = arg
    e = device_elf(o)
    if e is None: return os.path.relpath(o, build + '/' + T), {}
    with tempfile.NamedTemporaryFile(suffix='.elf', dir=os.environ.get('TMPDIR')) as f:
        f.write(e); f.flush()
        out = subprocess.run([LL + 'llvm-objdump', '-d', '--mcpu=gfx906', f.name], capture_output=True, text=True).stdout
    fn = {}; cur = None
    for l in out.splitlines():
        m = re.match(r'[0-9a-f]+ <(.+)>:$', l)
        if m: cur = fn.setdefault(m.group(1), []); continue
        m = re.match(r'\t(\S+)\s*(.*?)\s*// ([0-9A-F]+):', l)
        if m and cur is not None: cur.append((int(m.group(3), 16), m.group(1), m.group(2)))
    res = {}
    for name, ins in fn.items():
        addr = {a: i for i, (a, _, _) in enumerate(ins)}
        inloop = bytearray(len(ins)); nl = 0; big = 0
        for i, (a, mn, ops) in enumerate(ins):
            if mn.startswith('s_cbranch') or mn == 's_branch':
                m = re.match(r'(-?\d+)', ops)
                if not m: continue
                off = int(m.group(1))
                if off >= 0x8000: off -= 0x10000
                t = addr.get(a + 4 + 4 * off)
                if t is not None and t <= i:
                    nl += 1; big = max(big, i - t + 1)
                    for k in range(t, i + 1): inloop[k] = 1
        rl = sum(1 for i, (_, mn, _) in enumerate(ins) if inloop[i] and mn.startswith('buffer_load_'))
        sl = sum(1 for i, (_, mn, _) in enumerate(ins) if inloop[i] and mn.startswith('buffer_store_'))
        res[name] = (len(ins), nl, rl, sl, sum(1 for _, mn, _ in ins if mn.startswith('buffer_load_')), big)
    return os.path.relpath(o, build + '/' + T), res
def build(b, pool):
    objs = sorted(glob.glob(b + '/' + T + '/**/*.cu.o', recursive=True))
    return dict(pool.map(scan, [(b, o) for o in objs]))
if __name__ == '__main__':
    A, B = sys.argv[1], sys.argv[2]; lim = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    with multiprocessing.Pool(int(os.environ.get('JOBS', '8'))) as pool:
        a, b = build(A, pool), build(B, pool)
    rows = []; per = {}
    n = same = 0
    for rel in a:
        for name, x in a[rel].items():
            y = b.get(rel, {}).get(name)
            if y is None: continue
            n += 1; d = y[2] - x[2]
            if d == 0: same += 1
            t = per.setdefault(rel, [0, 0, 0, 0]); t[0] += d > 0; t[1] += d < 0; t[2] += max(d, 0); t[3] += min(d, 0)
            if abs(d) >= lim: rows.append((d, rel, name, x, y))
    more = sum(1 for r in rows if r[0] > 0); less = sum(1 for r in rows if r[0] < 0)
    print('%d objects, %d functions in both builds; reloads inside loops: more in B for %d, fewer for %d (threshold %d), unchanged for %d'
          % (len(a), n, more, less, lim, same))
    print('functions with any reload inside a loop: A %d, B %d' % (sum(1 for r in a.values() for x in r.values() if x[2]),
                                                                    sum(1 for r in b.values() for x in r.values() if x[2])))
    print('per object (functions with more / fewer, instructions added / removed):')
    for rel, t in sorted(per.items(), key=lambda kv: -(kv[1][2] - kv[1][3])):
        if t[0] or t[1]: print('  %-58s +%d / -%d functions, %+d / %+d reloads' % (rel.replace('template-instances/', 'ti/'), t[0], t[1], t[2], t[3]))
    print('by function (change; reloads in loops A -> B; spills in loops; instructions; largest loop):')
    for d, rel, name, x, y in sorted(rows, key=lambda r: (-r[0], r[1], r[2])):
        print('%+4d  reloads %3d -> %3d  spills %3d -> %3d  insns %5d -> %5d  loop %5d -> %5d  %-40s %s'
              % (d, x[2], y[2], x[3], y[3], x[0], y[0], x[5], y[5], rel.replace('template-instances/', 'ti/')[:40], re.sub(r'^_ZL?\d+', '', name)[:80]))
