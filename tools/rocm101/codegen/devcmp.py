#!/usr/bin/env python3
"""devcmp.py BUILD_A BUILD_B -- compare the gfx906 machine code (.text of the embedded code object) of every HIP object."""
import glob, hashlib, os, struct, subprocess, sys
T = 'ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda'
def text_hash(path):
    b = open(path, 'rb').read(); i = 0
    while True:
        i = b.find(b'\x7fELF', i + 1)
        if i < 0: return None
        if b[i + 4] == 2 and struct.unpack_from('<H', b, i + 18)[0] == 224:
            shoff, = struct.unpack_from('<Q', b, i + 40); es, n, strndx = struct.unpack_from('<HHH', b, i + 58)
            stroff, = struct.unpack_from('<Q', b, i + shoff + strndx * es + 24)
            h = hashlib.sha256()
            for k in range(n):
                name, typ = struct.unpack_from('<II', b, i + shoff + k * es)
                off, size = struct.unpack_from('<QQ', b, i + shoff + k * es + 24)
                nm = b[i + stroff + name:b.find(b'\0', i + stroff + name)]
                if nm == b'.text': h.update(b[i + off:i + off + size])
            return h.hexdigest()
a, b = sys.argv[1], sys.argv[2]
same = diff = 0; names = []
for o in sorted(glob.glob(a + '/' + T + '/**/*.cu.o', recursive=True)):
    rel = os.path.relpath(o, a + '/' + T)
    ha, hb = text_hash(o), text_hash(b + '/' + T + '/' + rel)
    if ha == hb: same += 1
    else: diff += 1; names.append(rel)
import collections
print("%d HIP objects: GPU code identical in %d, different in %d%s" % (same + diff, same, diff, (": " + ", ".join("%s x%d" % kv for kv in collections.Counter(n.split('-instance')[0].replace('template-instances/', '') for n in names).items())) if names else ""))
