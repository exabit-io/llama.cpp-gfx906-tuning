#!/usr/bin/env python3
"""tilescan.py OBJECT [OBJECT ...] -- host objects (or builds' objects) of ONE flash-attention tile instance: for every
flash_attn_tile<DKQ,DV,ncols1,ncols2,false> kernel the scratch reloads inside loops, one row per object. A per-object
compiler option changes every kernel of the object, and each batch size runs a different one (fattn-tile.cuh,
launch_fattn_tile_switch_ncols1: with ncols2 = 2, one query token runs <1,2>, two <2,2>, three or four <4,2>, five
to eight <8,2>, nine or more <16,2>), so an option chosen for the prefill kernel has to be read on the others too.
Columns are ncols1 x ncols2, grouped by ncols2; a value is reloads in loops / size of the largest loop."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loopscan import scan
rows = []; keys = set()
for o in sys.argv[1:]:
    _, r = scan((os.path.dirname(o), o))
    d = {}
    for name, x in r.items():
        m = re.search(r'flash_attn_tileILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELb0E', name)
        if m: d[(int(m.group(4)), int(m.group(3)))] = x
    rows.append((o, d)); keys |= set(d)
keys = sorted(keys)
print('%-28s' % 'object' + ''.join('%11s' % ('%dx%d' % (k[1], k[0])) for k in keys))
for o, d in rows:
    n = os.path.basename(o).replace('.cu.o', '').replace('try-', '').replace('.o', '')
    print('%-28s' % n[:28] + ''.join('%11s' % ('%d/%d' % (d[k][2], d[k][5]) if k in d else '-') for k in keys))
