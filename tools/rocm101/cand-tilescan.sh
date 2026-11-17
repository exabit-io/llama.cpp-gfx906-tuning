#!/bin/bash
# cand-tilescan.sh -- static reading of the candidates pool's head-256 tile object under each compiler (report s.7:
# gfx906-candidates 5c01f4ffd has no max-ilp default, no tile rule and its own head-256 tile-table row for <1,2>).
# Compiles ONE source, fattn-tile-instance-dkq256-dv256.cu of /root/wt-candidates-v100, with the command recorded in
# /root/build-candidates-v100/compile_commands.json, the compiler exchanged and options added per arm, then counts the
# scratch reloads inside loops of every tile kernel (codegen/tilescan.py). No GPU, no timing: it says where to look.
#   c23       ROCm 10.0's clang 23, the pool's flags (how /root/build-candidates-v100 was built)
#   c23ilp    clang 23 + max-ilp (master's scheduler)
#   c24       ROCm 10.1's clang 24, the pool's flags
#   c24t85    clang 24 + -amdgpu-vgpr-threshold-percent=85
#   c24ilp    clang 24 + max-ilp
#   c24ilp85  clang 24 + max-ilp + threshold 85 (master's flags since 224b73e24)
# READING, WRITTEN BEFORE THE FIRST COMPILE: an arm is "clean" if no tile kernel has more reloads in loops than the same
# kernel in c23. If c24 is not clean, the pool must not be compiled by clang 24 as it is. If c24t85 or c24ilp85 is
# clean, that is the candidate setting for a timing campaign (design points AND MTP at depth); a static count is not a
# timing result and decides nothing by itself (report s.5.12).
# Host compiles: run only when no timing run is in progress.
set -u
W=/root/rocm101-migration-20261114; O=$W/candtile; SRC=/root/wt-candidates-v100; B=/root/build-candidates-v100
mkdir -p $O; P=$O/progress
log(){ echo "$(date -Is) [candtile] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
[ -f $B/compile_commands.json ] || { log "FATAL: no $B/compile_commands.json"; exit 1; }
log "source $SRC at $(git -C $SRC rev-parse --short HEAD); recorded command from $B"
python3 -I - $B $O <<'EOF' 2>&1 | tee -a $P
import json, shlex, subprocess, sys, time
b, o = sys.argv[1:3]
cc = [c for c in json.load(open(b + '/compile_commands.json')) if c['file'].endswith('fattn-tile-instance-dkq256-dv256.cu')]
assert len(cc) == 1
args = shlex.split(cc[0]['command'])
if args[0].endswith('ccache'): args = args[1:]
assert not any('amdgpu-sched-strategy' in a or 'vgpr-threshold' in a for a in args), 'the recorded command already has a scheduler option'
ILP = ['-mllvm', '-amdgpu-sched-strategy=max-ilp']; T85 = ['-mllvm', '-amdgpu-vgpr-threshold-percent=85']
C23 = '/opt/rocm/core-10.0/lib/llvm/bin/clang++'; C24 = '/opt/rocm/core-10.1/lib/llvm/bin/clang++'
arms = [('c23', C23, []), ('c23ilp', C23, ILP), ('c24', C24, []), ('c24t85', C24, T85), ('c24ilp', C24, ILP), ('c24ilp85', C24, ILP + T85)]
for name, comp, extra in arms:
    out = '%s/%s.o' % (o, name); new = [comp]; skip = False
    for a in args[1:]:
        if skip: skip = False; continue
        if a == '-o': new += ['-o', out]; skip = True; continue
        new.append(a)
    new += extra
    t0 = time.time()
    p = subprocess.run(new, cwd=cc[0]['directory'], capture_output=True, text=True)
    print('  %-9s %s rc=%d %.0f s%s' % (name, ' '.join(extra) or '(pool flags)', p.returncode, time.time() - t0,
                                       '' if p.returncode == 0 else ' ' + (p.stderr.strip().splitlines() or ['?'])[-1][:160]))
EOF
objs=""; for n in c23 c23ilp c24 c24t85 c24ilp c24ilp85; do [ -s $O/$n.o ] && objs="$objs $O/$n.o"; done
[ -n "$objs" ] || { log "FATAL: nothing compiled"; exit 1; }
TMPDIR=$O python3 -I $W/codegen/tilescan.py $objs > $O/tilescan-candidates.txt 2>&1; log "tilescan rc=$?"
cat $O/tilescan-candidates.txt | tee -a $P
python3 -I - $O/tilescan-candidates.txt <<'EOF' 2>&1 | tee -a $P
import sys
rows = [l.split() for l in open(sys.argv[1]) if l.strip()]
if not rows or rows[0][0] != 'object': print('no table: NOT READ'); sys.exit(0)
d = {r[0]: [int(x.split('/')[0]) if '/' in x else None for x in r[1:]] for r in rows[1:]}
ref = d.get('c23')
if ref is None: print('no c23 row: NOT READ'); sys.exit(0)
for k, v in d.items():
    if k == 'c23': continue
    worse = [rows[0][1 + i] for i, (a, b) in enumerate(zip(v, ref)) if a is not None and b is not None and a > b]
    print('  %-9s %s' % (k, 'clean' if not worse else 'MORE reloads in loops than c23 in: ' + ' '.join(worse)))
EOF
log "=== CANDTILE DONE ==="
