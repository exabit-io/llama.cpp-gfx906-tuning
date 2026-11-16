#!/usr/bin/env python3
"""loopstat.py ELF [ELF ...] -- one line per gfx906 code object for the kernel matching KSUB (env; default the 16x2
head-256 tile kernel): size of its largest loop and what that loop does with memory. Scratch (spill) traffic shows as
buffer_load/buffer_store; data traffic as global_load/global_store. For ranking compiler variants before timing them."""
import os, re, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from asmcmp import insns, loops, sym
sub = os.environ.get('KSUB', 'flash_attn_tileILi256ELi256ELi16ELi2ELb0')
print('%-44s %6s %6s | hot loop: %5s %7s %7s %7s %6s %5s %5s' % ('code object', 'insns', 'loops', 'insns', 'gload', 'reload', 'spill', 'waits', 'lds', 'v_mov'))
for e in sys.argv[1:]:
    try:
        ins = insns(e, sym(e, sub))
    except SystemExit as x:
        print('%-44s %s' % (os.path.basename(e), x)); continue
    L = loops(ins)
    if not L:
        print('%-44s %6d %6d | no loop' % (os.path.basename(e), len(ins), 0)); continue
    s, t = max(L, key=lambda x: x[1] - x[0]); body = ins[s:t + 1]
    c = lambda rx: sum(1 for _, m, _ in body if re.match(rx, m))
    waits = sum(1 for _, m, o in body if m == 's_waitcnt' and 'vmcnt' in o)
    print('%-44s %6d %6d |           %5d %7d %7d %7d %6d %5d %5d' % (os.path.basename(e), len(ins), len(L), len(body),
          c(r'(global|flat)_load_'), c(r'buffer_load_'), c(r'buffer_store_'), waits, c(r'ds_read'), c(r'v_mov_b32$')))
