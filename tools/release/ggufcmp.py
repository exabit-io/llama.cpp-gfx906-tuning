import sys, hashlib
sys.path.insert(0, '/root/wt-master-v070/gguf-py')
from gguf import GGUFReader
ra, rb = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
ta = {t.name: t for t in ra.tensors}; tb = {t.name: t for t in rb.tensors}
same = diff = 0; diffs = []
for n in ta:
    ha = hashlib.md5(ta[n].data.tobytes()).hexdigest(); hb = hashlib.md5(tb[n].data.tobytes()).hexdigest()
    if ha == hb: same += 1
    else: diff += 1; diffs.append(n)
print(f"identical tensors {same}, different {diff}")
import collections
print("differing tensor kinds:", collections.Counter(n.split('.', 2)[-1] if n.startswith('blk.') else n for n in diffs).most_common(12))
