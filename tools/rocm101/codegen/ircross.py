#!/usr/bin/env python3
"""ircross.py NAME IRSTREAM CGSTREAM [CC1_FLAG ...] -- the head-256 flash-attention tile source with the optimizer of one
compiler and the code generator of the other. Step 1: the IRSTREAM compiler (100: clang 23, 101: clang 24) runs its
whole -O3 pipeline on the device code and stops at optimized bitcode (--cuda-device-only -emit-llvm, direct pipeline).
Step 2: the CGSTREAM compiler turns that bitcode into machine code without running its own optimizer (cc1 -x ir
-disable-llvm-passes at -O3), then its own driver jobs link the code object and embed it in the host object exactly as
a normal direct compile does (the job list comes from 'clang -###'). Step 3 as drvexp.py: /root/swap/NAME/libggml-hip.so
from the ROCm 10.1 build's objects with this one object replaced. IRSTREAM == CGSTREAM is the control for the method.
For telling whether clang 24's slower tile kernel comes from its optimizer or its code generator."""
import json, os, re, shlex, subprocess, sys
name, irs, cgs = sys.argv[1:4]; extra = sys.argv[4:]
BLD = {'100': '/root/build-master-v100r', '101': '/root/build-master-v101'}; B = '/root/build-master-v101'
TU = 'fattn-tile-instance-dkq256-dv256'
here = os.path.dirname(os.path.abspath(__file__))
def recorded(stream):
    cc = [c for c in json.load(open(BLD[stream] + '/compile_commands.json')) if c['file'].endswith(TU + '.cu')][0]
    a = shlex.split(cc['command'])
    if os.path.basename(a[0]) == 'ccache': a = a[1:]
    if stream == '101': a.append('--no-offload-new-driver')
    return a, cc['directory']
def fail(what, r):
    print(name, what, "FAILED:", (r.stderr.strip().splitlines() or ['?'])[-1][:260]); sys.exit(1)
# 1. optimized device bitcode from the IR compiler
a, d = recorded(irs); bc = '%s/irx-%s.bc' % (here, name)
a[a.index('-o') + 1] = bc
r = subprocess.run(a + ['--cuda-device-only', '-emit-llvm'], cwd=d, capture_output=True, text=True)
if r.returncode: fail('bitcode', r)
# 2. the code-generating compiler's own jobs, the first one reading the bitcode
a, d = recorded(cgs)
rel = 'ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/template-instances/%s.cu.o' % TU
donor = '/root/swap/donor-' + name; out = donor + '/' + rel
os.makedirs(os.path.dirname(out), exist_ok=True)
a[a.index('-o') + 1] = out
jobs = [shlex.split(l) for l in subprocess.run(a[:1] + ['-###'] + a[1:], cwd=d, capture_output=True, text=True).stderr.splitlines() if l.startswith(' "')]
if len(jobs) != 4 or '-cc1' not in jobs[0] or '-emit-obj' not in jobs[0]: print(name, "unexpected job list:", [j[0] for j in jobs]); sys.exit(1)
j = jobs[0]; k = []; i = 0
while i < len(j):
    if j[i] == '-mlink-builtin-bitcode': i += 2; continue        # the bitcode already holds the device library code it uses
    if j[i] == '-x' and j[i + 1] == 'hip': k += ['-x', 'ir', bc]; i += 3; continue
    k.append(j[i]); i += 1
assert bc in k, 'input not replaced'
jobs[0] = k + ['-disable-llvm-passes'] + extra
for n, j in enumerate(jobs):
    r = subprocess.run(j, cwd=d, capture_output=True, text=True)
    if r.returncode: fail('job %d (%s)' % (n + 1, os.path.basename(j[0])), r)
elf = here + '/drv-%s.elf' % name
subprocess.run([sys.executable, '-I', here + '/carve.py', out, elf], capture_output=True)
notes = subprocess.run(['/opt/rocm/core-10.1/lib/llvm/bin/llvm-readelf', '--notes', elf], capture_output=True, text=True).stdout
cur, res = None, []
for ln in notes.splitlines():
    m = re.match(r'\s+-?\s*\.(\w+):\s*(.*)', ln)
    if not m: continue
    key, v = m.group(1), m.group(2).strip()
    if key == 'name':
        cur = {'name': v} if re.search(r'flash_attn_tileILi256ELi256ELi(4ELi8|8ELi4|16ELi2)ELb0', v) else None
        if cur: res.append(cur)
    elif cur is not None and key in ('private_segment_fixed_size', 'vgpr_count', 'vgpr_spill_count'): cur[key] = int(v)
l = subprocess.run([sys.executable, '-I', here + '/swaplink.py', B, donor, '/root/swap/' + name, TU], capture_output=True, text=True)
print("%-12s optimizer clang %s, code generator clang %s %s\n   kernels: %s\n   %s" % (name, {'100': '23', '101': '24'}[irs], {'100': '23', '101': '24'}[cgs], ' '.join(extra),
      "; ".join("%s: %d spilled, %d vgprs, stack %d" % (re.sub(r'.*ILi256ELi256ELi(\d+)ELi(\d+).*', r'\1x\2', x['name']), x.get('vgpr_spill_count', 0), x.get('vgpr_count', 0), x.get('private_segment_fixed_size', 0)) for x in res),
      (l.stdout.strip() or l.stderr.strip()).splitlines()[-1][:200]))
