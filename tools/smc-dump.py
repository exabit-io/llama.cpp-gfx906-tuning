#!/usr/bin/env python3
# Dump every Apple SMC key via applesmc's key_at_index interface: name, type, length, raw hex, decoded value.
import struct, sys, time
S='/sys/devices/LNXSYSTM:00/LNXSYBUS:00/PNP0A08:00/device:1f/APP0001:00'
def rd(f, b=False):
    with open(f'{S}/{f}', 'rb' if b else 'r') as h: return h.read() if b else h.read().strip()
def decode(t, d):
    try:
        if t=='flt ' and len(d)==4: return f'{struct.unpack("<f",d)[0]:.3f}'
        if t=='sp78' and len(d)==2: return f'{struct.unpack(">h",d)[0]/256:.2f}'
        if t=='fp1f' and len(d)==2: return f'{struct.unpack(">H",d)[0]/32768:.3f}'
        if t=='fpe2' and len(d)==2: return f'{struct.unpack(">H",d)[0]/4:.2f}'
        if t=='ui8 ' and len(d)==1: return str(d[0])
        if t=='si8 ' and len(d)==1: return str(struct.unpack("b",d)[0])
        if t=='ui16' and len(d)==2: return str(struct.unpack(">H",d)[0])
        if t=='si16' and len(d)==2: return str(struct.unpack(">h",d)[0])
        if t=='ui32' and len(d)==4: return str(struct.unpack(">I",d)[0])
        if t=='flag' and len(d)==1: return str(d[0])
        if t=='ioft' and len(d)==8: return f'{struct.unpack(">Q",d)[0]/65536:.3f}'
        if t in ('ch8*','ch8 '): return repr(d.rstrip(b'\0').decode('ascii','replace'))
    except Exception as e: return f'?{e}'
    return ''
n=int(rd('key_count')); out=open(sys.argv[1],'w'); t0=time.time()
out.write(f'# SMC key dump {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  keys={n}\n# name type len hex decoded\n')
for i in range(n):
    try:
        with open(f'{S}/key_at_index','w') as h: h.write(str(i))
        name=rd('key_at_index_name'); typ=rd('key_at_index_type'); ln=rd('key_at_index_data_length'); d=rd('key_at_index_data',True)
        out.write(f'{name} {typ!r} {ln} {d.hex()} {decode(typ,d)}\n')
    except Exception as e: out.write(f'#{i} error {e}\n')
out.close(); print(f'{n} keys in {time.time()-t0:.1f}s -> {sys.argv[1]}')
