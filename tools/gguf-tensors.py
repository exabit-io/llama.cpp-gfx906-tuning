#!/usr/bin/env python3
"""gguf-tensors.py FILE...: list tensor bytes by name group without numpy (header read only)."""
import struct, sys, re, collections
BLK={0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),9:(32,36),10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292),16:(256,66),17:(256,50),18:(256,110),19:(256,82),20:(32,18),21:(256,32),22:(256,58),23:(256,90),24:(1,1),25:(1,2),26:(1,4),29:(256,88),30:(1,2)}
SZ={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
tot=collections.Counter(); cnt=collections.Counter(); types=collections.defaultdict(set); grand=0
for fn in sys.argv[1:]:
    f=open(fn,'rb'); magic,ver,n_t,n_kv=struct.unpack('<IIQQ',f.read(24))
    def rd(t):
        if t==8: n,=struct.unpack('<Q',f.read(8)); return f.read(n).decode('utf-8','replace')
        if t==9: et,n=struct.unpack('<IQ',f.read(12)); return [rd(et) for _ in range(n)]
        return struct.unpack('<'+SZ[t],f.read(struct.calcsize(SZ[t])))[0]
    for _ in range(n_kv):
        rd(8); t,=struct.unpack('<I',f.read(4)); rd(t)
    for _ in range(n_t):
        name=rd(8); nd,=struct.unpack('<I',f.read(4)); dims=struct.unpack('<'+'Q'*nd,f.read(8*nd)); ty,off=struct.unpack('<IQ',f.read(12))
        n=1
        for d in dims: n*=d
        bs,ts=BLK.get(ty,(1,4)); nbytes=n//bs*ts
        g=re.sub(r'\d+','N',name.split('.weight')[0]); g=re.sub(r'^blk\.N\.','blk.',g)
        tot[g]+=nbytes; cnt[g]+=1; types[g].add(ty); grand+=nbytes
for g,b in tot.most_common(40): print(f'{b/2**30:9.2f} GiB  {cnt[g]:5d}  types {sorted(types[g])}  {g}')
print(f'{grand/2**30:9.2f} GiB total')
