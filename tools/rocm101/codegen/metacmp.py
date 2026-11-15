#!/usr/bin/env python3
"""metacmp.py BUILD_A BUILD_B -- per gfx906 kernel: register counts, spills, stack and code size as the code objects record
them; how many kernels differ in any of them, and in which objects."""
import glob, os, re, struct, subprocess, sys
LL = '/opt/rocm/core-10.1/lib/llvm/bin/llvm-readelf'
T = 'ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda'
KEYS = ('private_segment_fixed_size', 'vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count', 'kernarg_segment_size')
def carve(path):
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
def kernels(build):
    out = {}
    tmp = '/tmp/claude-0/-root/c60c2d8f-3b04-4efe-9a97-141175b23c98/scratchpad/mc-%d.elf' % os.getpid()
    for o in glob.glob(build + '/' + T + '/**/*.cu.o', recursive=True):
        e = carve(o)
        if e is None: continue
        open(tmp, 'wb').write(e)
        rel = os.path.relpath(o, build + '/' + T)
        notes = subprocess.run([LL, '--notes', tmp], capture_output=True, text=True).stdout
        syms = subprocess.run([LL, '-s', '-W', tmp], capture_output=True, text=True).stdout
        size = {f[7]: int(f[2]) for f in (ln.split() for ln in syms.splitlines()) if len(f) >= 8 and f[3] == 'FUNC'}
        cur = None
        for ln in notes.splitlines():
            m = re.match(r'\s+-?\s*\.(\w+):\s*(.*)', ln)
            if not m: continue
            k, v = m.group(1), m.group(2).strip()
            if k == 'name':
                cur = out.setdefault((rel, v), {}); cur['code'] = size.get(v, 0)
            elif cur is not None and k in KEYS: cur[k] = int(v)
    os.remove(tmp)
    return out
a, b = kernels(sys.argv[1]), kernels(sys.argv[2])
common = [k for k in a if k in b]
diff = [k for k in common if a[k] != b[k]]
objs = {}
for rel, _ in diff: objs[rel] = objs.get(rel, 0) + 1
print("%d kernels: %d with identical register counts, spills, stack and code size; %d differ%s" % (len(common), len(common) - len(diff), len(diff), "" if not diff else " (in " + ", ".join("%s x%d" % (k.replace('template-instances/', ''), v) for k, v in sorted(objs.items())[:14]) + ")"))
