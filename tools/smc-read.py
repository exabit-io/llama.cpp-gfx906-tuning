#!/usr/bin/env python3
# Read named Apple SMC keys through applesmc's key_at_index interface (index map cached from smc-keys-*.txt dump).
# usage: smc-read.py KEY [KEY...]      prints "KEY=value" pairs on one line (flt/sp78/spc3/ui8/ui16/ui32/flag decoded)
import struct, sys, os, json
S='/sys/devices/LNXSYSTM:00/LNXSYBUS:00/PNP0A08:00/device:1f/APP0001:00'
MAP=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smc-keymap.json')
def build_map():
    m={}; n=int(open(f'{S}/key_count').read())
    for i in range(n):
        open(f'{S}/key_at_index','w').write(str(i)); m[open(f'{S}/key_at_index_name').read().strip()]=i
    json.dump(m, open(MAP,'w')); return m
m=json.load(open(MAP)) if os.path.exists(MAP) else build_map()
def dec(t,d):
    try:
        if t=='flt' and len(d)==4: return f'{struct.unpack("<f",d)[0]:.1f}'
        if t=='sp78': return f'{struct.unpack(">h",d)[0]/256:.1f}'
        if t=='spc3': return f'{struct.unpack(">h",d)[0]/8:.1f}'
        if t in('ui8','flag','si8'): return str(d[0])
        if t=='ui16': return str(struct.unpack(">H",d)[0])
        if t=='ui32': return str(struct.unpack(">I",d)[0])
    except Exception: pass
    return d.hex()
out=[]
for k in sys.argv[1:]:
    i=m.get(k)
    if i is None: m=build_map(); i=m.get(k)
    if i is None: out.append(f'{k}=?'); continue
    open(f'{S}/key_at_index','w').write(str(i))
    if open(f'{S}/key_at_index_name').read().strip()!=k: m=build_map(); open(f'{S}/key_at_index','w').write(str(m[k]))
    t=open(f'{S}/key_at_index_type').read().strip(); d=open(f'{S}/key_at_index_data','rb').read()
    out.append(f'{k}={dec(t,d)}')
print('  '.join(out))
