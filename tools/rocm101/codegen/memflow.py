#!/usr/bin/env python3
"""memflow.py ELF SYMBOL_SUBSTRING [ELF ...] -- memory traffic of one gfx906 function, loop by loop: how many loads and
stores each loop body issues, how many loads are in flight when the code waits for memory (s_waitcnt vmcnt), and where in
the body the loads sit. A kernel that waits after every load serialises memory latency; one that issues a burst and waits
once overlaps it. For comparing what two compilers scheduled."""
import collections, difflib, re, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from asmcmp import insns, loops, sym
LOAD = re.compile(r'^(global|flat|buffer)_load_'); STORE = re.compile(r'^(global|flat|buffer)_store_')
def flow(body):
    pend = 0; waits = []; nl = ns = 0; pos = []
    for i, (_, mn, ops) in enumerate(body):
        if LOAD.match(mn): pend += 1; nl += 1; pos.append(i)
        elif STORE.match(mn): ns += 1
        elif mn == 's_waitcnt':
            m = re.search(r'vmcnt\((\d+)\)', ops)
            if m:
                k = int(m.group(1)); waits.append((pend, k)); pend = min(pend, k)
    return nl, ns, waits, pos
sub = sys.argv[2]; elfs = [sys.argv[1]] + sys.argv[3:]
seqs = []
for e in elfs:
    ins = insns(e, sym(e, sub)); L = sorted(loops(ins), key=lambda x: x[1] - x[0])
    print(f'== {e}: {len(ins)} instructions, {len(L)} loops')
    for s, t in L:
        body = ins[s:t + 1]; nl, ns, waits, pos = flow(body)
        h = collections.Counter(p for p, _ in waits)
        inner = sum(1 for a, b in L if s <= a and b <= t) - 1
        print(f'  loop of {len(body):5d} insns (holds {inner} loops): loads {nl:3d}, stores {ns:3d}, memory waits {len(waits):3d}, '
              f'loads in flight at a wait: max {max((p for p, _ in waits), default=0)}, mean {sum(p for p, _ in waits) / max(1, len(waits)):.2f}; '
              f'histogram {dict(sorted(h.items()))}')
    s, t = max(L, key=lambda x: x[1] - x[0]); seqs.append([m for _, m, _ in ins[s:t + 1]])
if len(seqs) == 2:
    sm = difflib.SequenceMatcher(None, seqs[0], seqs[1], autojunk=False)
    print(f'largest loop, mnemonic sequences: similarity {sm.ratio():.3f}; equal runs {sum(1 for o in sm.get_opcodes() if o[0] == "equal")}, '
          f'changed runs {sum(1 for o in sm.get_opcodes() if o[0] != "equal")}')
