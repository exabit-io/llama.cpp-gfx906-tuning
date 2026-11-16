#!/usr/bin/env python3
"""drvexp.py NAME STREAM [--drop REGEX] [EXTRA_FLAG ...] -- compile the head-256 flash-attention tile source with the recorded command
of the ROCm STREAM build (100: /root/build-master-v100r, clang 23; 101: /root/build-master-v101, clang 24) plus the extra
flags (--drop removes recorded arguments that match, with the -mllvm in front of them), and link a libggml-hip.so (/root/swap/NAME) from the ROCm 10.1 build's objects with this one object replaced.
Prints the device pipeline the driver chose (direct object code or bitcode for link-time optimization) and the kernels'
spill counts. For telling what in the ROCm 10.1 toolchain slows this kernel: the compiler itself or the way its driver
compiles device code (the new offload driver sends it through full LTO; --no-offload-new-driver / -fno-offload-lto do not)."""
import json, os, re, shlex, subprocess, sys
name, stream = sys.argv[1:3]; extra = sys.argv[3:]
SRCB = {'100': '/root/build-master-v100r', '101': '/root/build-master-v101'}[stream]; B = '/root/build-master-v101'
TU = 'fattn-tile-instance-dkq256-dv256'
cc = [c for c in json.load(open(SRCB + '/compile_commands.json')) if c['file'].endswith(TU + '.cu')][0]
args = shlex.split(cc['command'])
if os.path.basename(args[0]) == 'ccache': args = args[1:]
rel = 'ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/template-instances/%s.cu.o' % TU
donor = '/root/swap/donor-' + name; out = donor + '/' + rel
os.makedirs(os.path.dirname(out), exist_ok=True)
args[args.index('-o') + 1] = out
if extra and extra[0] == '--drop':
    rx = re.compile(extra[1]); extra = extra[2:]; kept = []
    for a in args:
        if rx.search(a):
            if kept and kept[-1] == '-mllvm': kept.pop()
            continue
        kept.append(a)
    args = kept
args += extra
d = subprocess.run(args[:1] + ['-###'] + args[1:], cwd=cc['directory'], capture_output=True, text=True).stderr
dev = [shlex.split(l) for l in d.splitlines() if l.startswith(' "') and '"-cc1"' in l and '"amdgcn-amd-amdhsa"' in l.split('"-aux-triple"')[0]]
pipe = '?'
if dev:
    w = dev[0]
    pipe = ' '.join(x for x in w if x.startswith(('-emit', '-flto'))) + ' | ' + ' '.join(w[i + 1] for i, x in enumerate(w) if x == '-mllvm')
r = subprocess.run(args, cwd=cc['directory'], capture_output=True, text=True)
if r.returncode:
    print(name, "COMPILE FAILED:", (r.stderr.strip().splitlines() or ['?'])[-1][:240]); sys.exit(1)
here = os.path.dirname(os.path.abspath(__file__))
elf = here + '/drv-%s.elf' % name
subprocess.run([sys.executable, '-I', here + '/carve.py', out, elf], capture_output=True)
notes = subprocess.run(['/opt/rocm/core-10.1/lib/llvm/bin/llvm-readelf', '--notes', elf], capture_output=True, text=True).stdout
cur, res = None, []
for ln in notes.splitlines():
    m = re.match(r'\s+-?\s*\.(\w+):\s*(.*)', ln)
    if not m: continue
    k, v = m.group(1), m.group(2).strip()
    if k == 'name':
        cur = {'name': v} if re.search(r'flash_attn_tileILi256ELi256ELi(4ELi8|8ELi4|16ELi2)ELb0', v) else None
        if cur: res.append(cur)
    elif cur is not None and k in ('private_segment_fixed_size', 'vgpr_count', 'vgpr_spill_count'): cur[k] = int(v)
l = subprocess.run([sys.executable, '-I', here + '/swaplink.py', B, donor, '/root/swap/' + name, TU], capture_output=True, text=True)
print("%-12s clang %s %s\n   device step: %s\n   kernels: %s\n   %s" % (name, {'100': '23', '101': '24'}[stream], ' '.join(extra), pipe,
      "; ".join("%s: %d spilled, %d vgprs, stack %d" % (re.sub(r'.*ILi256ELi256ELi(\d+)ELi(\d+).*', r'\1x\2', x['name']), x.get('vgpr_spill_count', 0), x.get('vgpr_count', 0), x.get('private_segment_fixed_size', 0)) for x in res),
      (l.stdout.strip() or l.stderr.strip()).splitlines()[-1][:200]))
