#!/usr/bin/env python3
"""asmcmp.py ELF_A ELF_B SYMBOL_SUBSTRING [TOP] -- one gfx906 function in two code objects: instruction counts by
mnemonic (those that differ), backward branches (loops), and the mnemonics inside each loop body, innermost first.
For reading what a newer compiler changed in a kernel. Uses the ROCm 10.1 llvm-objdump for both."""
import collections, re, subprocess, sys
LL = '/opt/rocm/core-10.1/lib/llvm/bin/'
def sym(elf, sub):
    out = subprocess.run([LL + 'llvm-readelf', '-s', '-W', elf], capture_output=True, text=True).stdout
    names = sorted({l.split()[7] for l in out.splitlines() if len(l.split()) >= 8 and l.split()[3] == 'FUNC' and sub in l.split()[7]})
    if len(names) != 1: sys.exit(f'{elf}: {len(names)} functions match {sub!r}: {names[:4]}')
    return names[0]
def insns(elf, name):
    out = subprocess.run([LL + 'llvm-objdump', '-d', '--mcpu=gfx906', f'--disassemble-symbols={name}', elf],
                         capture_output=True, text=True).stdout
    r = []
    for l in out.splitlines():      # "\tmnemonic operands   // ADDRESS: raw words"
        m = re.match(r'\t(\S+)\s*(.*?)\s*// ([0-9A-F]+):', l)
        if m: r.append((int(m.group(3), 16), m.group(1), m.group(2)))
    return r
def loops(ins):
    """backward branches as (index of target, index of branch); the operand is a signed count of words after the branch"""
    addr = {a: i for i, (a, _, _) in enumerate(ins)}
    L = []
    for i, (a, mn, ops) in enumerate(ins):
        if mn.startswith('s_cbranch') or mn == 's_branch':
            m = re.match(r'(-?\d+)', ops)
            if not m: continue
            off = int(m.group(1))
            if off >= 0x8000: off -= 0x10000
            t = addr.get(a + 4 + 4 * off)
            if t is not None and t <= i: L.append((t, i))
    return L
def main():
    ea, eb, sub = sys.argv[1:4]; top = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    A, B = insns(ea, sym(ea, sub)), insns(eb, sym(eb, sub))
    ca, cb = collections.Counter(m for _, m, _ in A), collections.Counter(m for _, m, _ in B)
    print(f'instructions: {len(A)} -> {len(B)}; distinct mnemonics {len(ca)} -> {len(cb)}')
    rows = sorted(((cb[k] - ca[k], k) for k in set(ca) | set(cb) if ca[k] != cb[k]), key=lambda x: -abs(x[0]))
    for d, k in rows[:top]: print(f'  {k:28s} {ca[k]:5d} -> {cb[k]:5d}  ({d:+d})')
    for name, ins in (('A', A), ('B', B)):
        L = sorted(loops(ins), key=lambda x: x[1] - x[0])
        print(f'{name}: {len(L)} backward branches; loop bodies (instructions): {[e - s + 1 for s, e in L]}')
        for s, e in L[:3]:
            c = collections.Counter(m for _, m, _ in ins[s:e + 1])
            print(f'   body {e - s + 1}: ' + ', '.join(f'{k} {v}' for k, v in c.most_common(14)))
if __name__ == '__main__':
    main()
