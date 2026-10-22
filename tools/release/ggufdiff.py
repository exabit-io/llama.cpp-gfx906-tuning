import sys
sys.path.insert(0, '/root/wt-master-v070/gguf-py')
from gguf import GGUFReader
def info(p):
    r = GGUFReader(p)
    kv = {}
    for f in r.fields.values():
        if f.name.startswith('tokenizer.ggml.') and f.name not in ('tokenizer.ggml.model', 'tokenizer.ggml.pre'):
            continue
        try:
            v = f.contents()
        except Exception:
            v = '?'
        if isinstance(v, list) and len(v) > 8:
            v = f'list[{len(v)}]'
        kv[f.name] = v
    t = {x.name: (x.tensor_type.name, tuple(int(d) for d in x.shape)) for x in r.tensors}
    return kv, t
a, ta = info(sys.argv[1]); b, tb = info(sys.argv[2])
print("tensors", len(ta), len(tb), "nextn in A", sum('nextn' in k for k in ta), "in B", sum('nextn' in k for k in tb))
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        print("KV diff", k, "|", a.get(k), "|", b.get(k))
for k in sorted(set(ta) | set(tb)):
    if ta.get(k) != tb.get(k):
        print("T diff", k, ta.get(k), tb.get(k))
print("A arch", a.get('general.architecture'), "block_count", {k: v for k, v in a.items() if k.endswith('block_count') or 'nextn' in k})
