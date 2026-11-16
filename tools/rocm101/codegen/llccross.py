#!/usr/bin/env python3
"""llccross.py NAME DEVICE_OBJECT -- a libggml-hip.so whose head-256 flash-attention tile object carries machine code
produced OUTSIDE the ROCm compilers: DEVICE_OBJECT is a relocatable amdgcn object of that translation unit (llc output
from the bitcode codegen/irx-x23on23.bc, e.g. the per-commit objects of llvm-bisect/step.sh). ROCm 10.0's own driver
jobs for the source file are taken from 'clang -###' (direct pipeline: device cc1, lld, host cc1 with the embedded code
object); the first job is NOT run, DEVICE_OBJECT is put where its output would be, and the remaining jobs link the code
object and embed it in the host object exactly as a normal compile does. Then, as ircross.py: /root/swap/NAME/
libggml-hip.so from the ROCm 10.1 build's objects with this one object replaced.
Control for the method: llc built at ROCm 10.0's compiler commit must reproduce clang 23's object (same .text hash as
/root/swap/donor-x23on23's)."""
import hashlib, json, os, re, shlex, shutil, struct, subprocess, sys
name, devobj = sys.argv[1:3]
SRCB = '/root/build-master-v100r'; B = '/root/build-master-v101'
TU = 'fattn-tile-instance-dkq256-dv256'
here = os.path.dirname(os.path.abspath(__file__))
cc = [c for c in json.load(open(SRCB + '/compile_commands.json')) if c['file'].endswith(TU + '.cu')][0]
a = shlex.split(cc['command']); d = cc['directory']
if os.path.basename(a[0]) == 'ccache': a = a[1:]
rel = 'ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/template-instances/%s.cu.o' % TU
donor = '/root/swap/donor-' + name; out = donor + '/' + rel
os.makedirs(os.path.dirname(out), exist_ok=True)
a[a.index('-o') + 1] = out
jobs = [shlex.split(l) for l in subprocess.run(a[:1] + ['-###'] + a[1:], cwd=d, capture_output=True, text=True).stderr.splitlines() if l.startswith(' "')]
if len(jobs) != 4 or '-cc1' not in jobs[0] or '-emit-obj' not in jobs[0]: print(name, "unexpected job list:", [j[0] for j in jobs]); sys.exit(1)
tmpobj = jobs[0][jobs[0].index('-o') + 1]
if not any(tmpobj in j for j in jobs[1:]): print(name, "the device object of job 1 is not an input of the later jobs"); sys.exit(1)
shutil.copyfile(devobj, tmpobj)
for n, j in enumerate(jobs[1:], 2):
    r = subprocess.run(j, cwd=d, capture_output=True, text=True)
    if r.returncode: print(name, 'job %d (%s) FAILED:' % (n, os.path.basename(j[0])), (r.stderr.strip().splitlines() or ['?'])[-1][:260]); sys.exit(1)
os.unlink(tmpobj)
def text_hash(path):                      # devcmp.py's: .text of the embedded gfx906 code object
    b = open(path, 'rb').read(); i = 0
    while True:
        i = b.find(b'\x7fELF', i + 1)
        if i < 0: return None
        if b[i + 4] == 2 and struct.unpack_from('<H', b, i + 18)[0] == 224:
            shoff, = struct.unpack_from('<Q', b, i + 40); es, n, strndx = struct.unpack_from('<HHH', b, i + 58)
            stroff, = struct.unpack_from('<Q', b, i + shoff + strndx * es + 24)
            h = hashlib.sha256()
            for k in range(n):
                nm, typ = struct.unpack_from('<II', b, i + shoff + k * es)
                off, size = struct.unpack_from('<QQ', b, i + shoff + k * es + 24)
                if b[i + stroff + nm:b.find(b'\0', i + stroff + nm)] == b'.text': h.update(b[i + off:i + off + size])
            return h.hexdigest()
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
h = text_hash(out); ref = {}
for k in ('x23on23', 'x23on24', 'c23', 'c24'):
    p = '/root/swap/donor-%s/%s' % (k, rel)
    if os.path.exists(p): ref[k] = text_hash(p)
same = [k for k, v in ref.items() if v == h]
ls = subprocess.run([sys.executable, '-I', here + '/loopstat.py', elf], capture_output=True, text=True).stdout.strip().splitlines()
print("%-10s device object %s\n   kernels: %s\n   hot loop: %s\n   .text identical to: %s\n   %s" % (name, devobj,
      "; ".join("%s: %d spilled, %d vgprs, stack %d" % (re.sub(r'.*ILi256ELi256ELi(\d+)ELi(\d+).*', r'\1x\2', x['name']), x.get('vgpr_spill_count', 0), x.get('vgpr_count', 0), x.get('private_segment_fixed_size', 0)) for x in res),
      (ls[-1].split('|')[-1].strip() if ls else '?'), ', '.join(same) or 'none of ' + ', '.join(ref),
      (l.stdout.strip() or l.stderr.strip() or '?').splitlines()[-1][:200]))
