#!/usr/bin/env python3
"""trycc.py BUILD SRC_SUFFIX TAG [--drop REGEX] [EXTRA_FLAG ...] -- recompile one HIP source of a build with its recorded
command (minus arguments matching --drop, plus the extra flags), carve the gfx906 code object and print stack bytes /
spilled VGPRs / code bytes of the kernels matching KPAT (env, default: the three 32-column head-256 tile kernels)."""
import json, os, re, shlex, subprocess, sys, time
build, suffix, tag = sys.argv[1:4]
rest = sys.argv[4:]
drop = None
if rest and rest[0] == '--drop':
    drop = re.compile(rest[1]); rest = rest[2:]
cc = [c for c in json.load(open(build + '/compile_commands.json')) if c['file'].endswith(suffix)]
assert len(cc) == 1, [c['file'] for c in cc]
args = shlex.split(cc[0]['command'])
if os.path.basename(args[0]) == 'ccache':
    args = args[1:]
out = os.path.abspath('try-%s.o' % tag)
new, skip = [], False
for i, a in enumerate(args):
    if skip:
        skip = False; continue
    if a == '-o':
        new += ['-o', out]; skip = True; continue
    if drop is not None and a == '-mllvm' and i + 1 < len(args) and drop.search(args[i + 1]):
        skip = True; continue
    if drop is not None and drop.search(a):
        continue
    new.append(a)
new += rest
t0 = time.time()
p = subprocess.run(new, cwd=cc[0]['directory'], capture_output=True, text=True)
if p.returncode:
    print(tag, "COMPILE FAILED:", p.stderr.strip().splitlines()[-1][:200]); sys.exit(1)
subprocess.run([sys.executable, '-I', 'carve.py', out, 'try-%s.elf' % tag], capture_output=True)
notes = subprocess.run(['/opt/rocm/core-10.1/lib/llvm/bin/llvm-readelf', '--notes', 'try-%s.elf' % tag], capture_output=True, text=True).stdout
kpat = re.compile(os.environ.get('KPAT', r'flash_attn_tileILi256ELi256ELi(4ELi8|8ELi4|16ELi2)ELb0'))
cur, res = None, []
for ln in notes.splitlines():
    m = re.match(r'\s+-?\s*\.(\w+):\s*(.*)', ln)
    if not m:
        continue
    k, v = m.group(1), m.group(2).strip()
    if k == 'name':
        cur = {'name': v} if kpat.search(v) else None
        if cur: res.append(cur)
    elif cur is not None and k in ('private_segment_fixed_size', 'vgpr_count', 'vgpr_spill_count'):
        cur[k] = int(v)
warn = sum(1 for l in p.stderr.splitlines() if 'warning' in l)
print("%-34s %s | %d warnings, %.0f s" % (tag, "; ".join("%s: %d spilled, stack %d, vgprs %d" % (re.sub(r'.*ILi256ELi256ELi(\d+)ELi(\d+).*', r'\1x\2', r['name']), r.get('vgpr_spill_count', 0), r.get('private_segment_fixed_size', 0), r.get('vgpr_count', 0)) for r in res), warn, time.time() - t0))
