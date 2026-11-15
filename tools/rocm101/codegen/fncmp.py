#!/usr/bin/env python3
"""fncmp.py BUILD_A BUILD_B [REGEX] -- per gfx906 function (by symbol name): are the machine-code bytes the same in the
two builds? Independent of the order in which the functions are laid out."""
import glob, hashlib, os, re, struct, sys
T = 'ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda'
def funcs(path):
    b = open(path, 'rb').read(); i = 0
    while True:
        i = b.find(b'\x7fELF', i + 1)
        if i < 0: return {}
        if b[i + 4] == 2 and struct.unpack_from('<H', b, i + 18)[0] == 224: break
    e = i
    shoff, = struct.unpack_from('<Q', b, e + 40); es, n, strndx = struct.unpack_from('<HHH', b, e + 58)
    secs = []
    for k in range(n):
        name, typ = struct.unpack_from('<II', b, e + shoff + k * es)
        addr, off, size = struct.unpack_from('<QQQ', b, e + shoff + k * es + 16)
        link, = struct.unpack_from('<I', b, e + shoff + k * es + 40)
        secs.append((typ, addr, off, size, link))
    out = {}
    for typ, addr, off, size, link in secs:
        if typ != 2: continue                       # SHT_SYMTAB
        stroff = secs[link][2]
        for s in range(size // 24):
            nm, info, other, shndx, val, sz = struct.unpack_from('<IBBHQQ', b, e + off + s * 24)
            if info & 0xf != 2 or sz == 0 or shndx == 0 or shndx >= len(secs): continue
            name = b[e + stroff + nm:b.find(b'\0', e + stroff + nm)].decode()
            st = secs[shndx]
            code = b[e + st[2] + (val - st[1]):e + st[2] + (val - st[1]) + sz]
            out[name] = (sz, hashlib.sha256(code).hexdigest())
    return out
a, b = sys.argv[1], sys.argv[2]; pat = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
tot = same = 0; diffs = {}
for o in sorted(glob.glob(a + '/' + T + '/**/*.cu.o', recursive=True)):
    rel = os.path.relpath(o, a + '/' + T)
    fa, fb = funcs(o), funcs(b + '/' + T + '/' + rel)
    for name in fa:
        if pat and not pat.search(name): continue
        if name not in fb: continue
        tot += 1
        if fa[name] == fb[name]: same += 1
        else: diffs[rel] = diffs.get(rel, 0) + 1
print("%d functions compared: %d byte-identical, %d different%s" % (tot, same, tot - same, "" if not diffs else " (in: " + ", ".join("%s x%d" % kv for kv in sorted(diffs.items(), key=lambda kv: -kv[1])[:6]) + (" ..." if len(diffs) > 6 else "") + ")"))
