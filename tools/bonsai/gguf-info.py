#!/usr/bin/env python3
# gguf-info.py FILE -- read a GGUF header (metadata and tensor table only, not the weights): the tensor types with
# their counts and bytes, and the metadata keys that say how the runtime must treat the weights.
import struct, sys, collections
f = open(sys.argv[1], 'rb')
def rd(fmt):
    n = struct.calcsize(fmt); return struct.unpack(fmt, f.read(n))[0]
def rstr():
    n = rd('<Q'); return f.read(n).decode('utf-8', 'replace')
T = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
def rval(t):
    if t in T: return rd(T[t])
    if t == 8: return rstr()
    if t == 9:
        et = rd('<I'); n = rd('<Q')
        if et in T:
            sz = struct.calcsize(T[et]); data = f.read(sz * n)
            return ('array', et, n, [struct.unpack_from(T[et], data, i * sz)[0] for i in range(min(n, 8))])
        vals = [rval(et) for _ in range(n)]
        return ('array', et, n, vals[:8])
    raise ValueError(t)
assert f.read(4) == b'GGUF'
ver = rd('<I'); nt = rd('<Q'); nk = rd('<Q')
kv = collections.OrderedDict()
for _ in range(nk):
    k = rstr(); t = rd('<I'); kv[k] = rval(t)
names = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',30:'BF16',34:'TQ1_0',35:'TQ2_0',41:'Q1_0',42:'Q2_0'}
tens = []
for _ in range(nt):
    name = rstr(); nd = rd('<I'); dims = [rd('<Q') for _ in range(nd)]; ty = rd('<I'); off = rd('<Q'); tens.append((name, dims, ty, off))
align = kv.get('general.alignment', 32)
pos = f.tell(); data0 = (pos + align - 1) // align * align
import os
size = os.path.getsize(sys.argv[1])
tens.sort(key=lambda x: x[3])
by = collections.defaultdict(lambda: [0, 0, 0])
for i, (name, dims, ty, off) in enumerate(tens):
    nxt = tens[i+1][3] if i + 1 < len(tens) else size - data0
    n = 1
    for d in dims: n *= d
    b = by[names.get(ty, 'type%d' % ty)]; b[0] += 1; b[1] += n; b[2] += nxt - off
print('GGUF v%d, %d tensors, %d metadata keys, file %.2f GiB' % (ver, nt, nk, size / 2**30))
for k in sorted(by, key=lambda k: -by[k][2]):
    c, n, b = by[k]; print('  %-6s %4d tensors  %7.3f G weights  %6.3f GiB  %.2f bits per weight' % (k, c, n / 1e9, b / 2**30, b * 8 / n))
for k, v in kv.items():
    if any(s in k for s in ('general.architecture', 'general.name', 'general.file_type', 'general.quantization', 'prism', 'hadamard', 'context_length', 'block_count', 'embedding_length', 'general.size_label', 'general.base_model.0.name')):
        s = repr(v); print('  %s = %s' % (k, s if len(s) < 160 else s[:160] + '...'))
big = sorted(tens, key=lambda x: -(x[1][0] * (x[1][1] if len(x[1]) > 1 else 1)))[:0]
ex = collections.OrderedDict()
for name, dims, ty, off in tens:
    key = names.get(ty, str(ty))
    if key not in ex or len(ex[key]) < 3: ex.setdefault(key, []).append('%s %s' % (name, 'x'.join(map(str, dims))))
for k, v in ex.items(): print('  e.g. %s: %s' % (k, '; '.join(v)))
