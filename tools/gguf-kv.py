#!/usr/bin/env python3
"""gguf-kv.py FILE [KEY-SUFFIX ...]: print scalar GGUF metadata without numpy (default: embedding_length, block_count)."""
import struct, sys
f = open(sys.argv[1], 'rb'); want = sys.argv[2:] or ['embedding_length', 'block_count']
magic, ver, n_t, n_kv = struct.unpack('<IIQQ', f.read(24)); assert magic == 0x46554747, 'not GGUF'
SZ = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def rd(t):
    if t == 8: n, = struct.unpack('<Q', f.read(8)); return f.read(n).decode('utf-8', 'replace')
    if t == 9: et, n = struct.unpack('<IQ', f.read(12)); return [rd(et) for _ in range(n)]
    return struct.unpack('<' + SZ[t], f.read(struct.calcsize(SZ[t])))[0]
for _ in range(n_kv):
    key = rd(8); t, = struct.unpack('<I', f.read(4)); v = rd(t)
    if any(key.endswith(w) for w in want) and not isinstance(v, list): print(key, v)
