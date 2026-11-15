#!/usr/bin/env python3
"""tileexp.py NAME NTHREADS OCCUPANCY NBATCH_FA NBATCH_K [EXTRA_FLAG ...] -- the head-256, 32-column entry of the AMD (non-RDNA)
flash-attention tile table with other values: edit a scratch worktree (/root/wt-tile-exp, reset first), compile the one
source with the ROCm 10.1 command of /root/build-master-v101, report the three 32-column kernels' spills, and link a
libggml-hip.so (/root/swap/NAME) from the 10.1 build's objects with this one object replaced."""
import json, os, re, shlex, subprocess, sys
name, nthr, occ, nfa, nk = sys.argv[1:6]; extra = sys.argv[6:]
WT = '/root/wt-tile-exp'; SRC = '/root/wt-master-v100'; B = '/root/build-tile-max-ilp'
subprocess.run(['git', '-C', WT, 'checkout', '-q', '--', '.'], check=True)
p = WT + '/ggml/src/ggml-cuda/fattn-tile.cuh'
lines = open(p).read().split('\n')
i0 = next(i for i, l in enumerate(lines) if 'ggml_cuda_fattn_tile_get_config_amd(const int DKQ' in l)
i = next(i for i in range(i0, len(lines)) if re.search(r'CONFIG_CASE\(256, 256, 32,', lines[i]))
old = lines[i]
lines[i] = '    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, %s, %s, %s, %s)' % (nthr, occ, nfa, nk)
open(p, 'w').write('\n'.join(lines))
cc = [c for c in json.load(open('/root/build-master-v101/compile_commands.json')) if c['file'].endswith('fattn-tile-instance-dkq256-dv256.cu')][0]
args = [a.replace(SRC, WT) for a in shlex.split(cc['command'])]
if os.path.basename(args[0]) == 'ccache': args = args[1:]
rel = 'ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/template-instances/fattn-tile-instance-dkq256-dv256.cu.o'
donor = '/root/swap/donor-' + name
out = donor + '/' + rel
os.makedirs(os.path.dirname(out), exist_ok=True)
args[args.index('-o') + 1] = out
r = subprocess.run(args + extra, cwd=cc['directory'], capture_output=True, text=True)
if r.returncode:
    print(name, "COMPILE FAILED:", (r.stderr.strip().splitlines() or ['?'])[-1][:240]); sys.exit(1)
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, '-I', here + '/carve.py', out, '/root/swap/donor-%s.elf' % name], capture_output=True)
notes = subprocess.run(['/opt/rocm/core-10.1/lib/llvm/bin/llvm-readelf', '--notes', '/root/swap/donor-%s.elf' % name], capture_output=True, text=True).stdout
cur, res = None, []
for ln in notes.splitlines():
    m = re.match(r'\s+-?\s*\.(\w+):\s*(.*)', ln)
    if not m: continue
    k, v = m.group(1), m.group(2).strip()
    if k == 'name':
        cur = {'name': v} if re.search(r'flash_attn_tileILi256ELi256ELi(4ELi8|8ELi4|16ELi2)ELb0', v) else None
        if cur: res.append(cur)
    elif cur is not None and k in ('private_segment_fixed_size', 'vgpr_count', 'vgpr_spill_count'): cur[k] = int(v)
l = subprocess.run([sys.executable, '-I', here + '/swaplink.py', B, donor, '/root/swap/' + name, 'fattn-tile-instance-dkq256-dv256'], capture_output=True, text=True)
print("%-12s (%s,%s,%s,%s)%s | %s | %s" % (name, nthr, occ, nfa, nk, (' ' + ' '.join(extra)) if extra else '', "; ".join("%s: %d spilled, %d vgprs" % (re.sub(r'.*ILi256ELi256ELi(\d+)ELi(\d+).*', r'\1x\2', x['name']), x.get('vgpr_spill_count', 0), x.get('vgpr_count', 0)) for x in res), 'linked' if l.returncode == 0 else 'LINK FAILED ' + l.stdout[-200:]))
subprocess.run(['git', '-C', WT, 'checkout', '-q', '--', '.'], check=True)
