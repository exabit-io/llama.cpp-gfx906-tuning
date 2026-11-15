#!/usr/bin/env python3
"""spilldiff.py BUILD_A BUILD_B -- every gfx906 kernel of the two builds' HIP objects: where the number of spilled vector
registers (or the stack) differs. Prints the kernels sorted by the change in spilled VGPRs."""
import glob, os, re, struct, subprocess, sys
LL = '/opt/rocm/core-10.1/lib/llvm/bin/llvm-readelf'
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
def kernels(build):
    out = {}
    for o in glob.glob(build + '/' + T + '/**/*.cu.o', recursive=True):
        e = device_elf(o)
        if e is None: continue
        tmp = '/tmp/claude-0/-root/c60c2d8f-3b04-4efe-9a97-141175b23c98/scratchpad/sd-%d.elf' % os.getpid()
        open(tmp, 'wb').write(e)
        notes = subprocess.run([LL, '--notes', tmp], capture_output=True, text=True).stdout
        cur = None; rel = os.path.relpath(o, build + '/' + T)
        for ln in notes.splitlines():
            m = re.match(r'\s+-?\s*\.(\w+):\s*(.*)', ln)
            if not m: continue
            k, v = m.group(1), m.group(2).strip()
            if k == 'name': cur = out.setdefault((rel, v), {})
            elif cur is not None and k in ('private_segment_fixed_size', 'vgpr_count', 'vgpr_spill_count', 'sgpr_spill_count'): cur[k] = int(v)
        os.remove(tmp)
    return out
a, b = kernels(sys.argv[1]), kernels(sys.argv[2])
rows = []
for key in a:
    if key in b:
        da = a[key].get('vgpr_spill_count', 0); db = b[key].get('vgpr_spill_count', 0)
        rows.append((db - da, key, da, db, a[key].get('private_segment_fixed_size', 0), b[key].get('private_segment_fixed_size', 0), a[key].get('vgpr_count', 0), b[key].get('vgpr_count', 0)))
print("%d kernels in both builds; spilled VGPRs: more in B for %d kernels, fewer for %d, same for %d" % (len(rows), sum(r[0] > 0 for r in rows), sum(r[0] < 0 for r in rows), sum(r[0] == 0 for r in rows)))
lim = int(sys.argv[3]) if len(sys.argv) > 3 else 4
for d, (rel, name), da, db, sa, sb, va, vb in sorted(rows, key=lambda r: -abs(r[0])):
    if abs(d) < lim: break
    print("%+4d  spilled %2d -> %2d  stack %3d -> %3d  vgprs %3d -> %3d  %-44s %s" % (d, da, db, sa, sb, va, vb, rel.replace('template-instances/', 'ti/')[:44], re.sub(r'^_ZL?\d+', '', name)[:70]))
