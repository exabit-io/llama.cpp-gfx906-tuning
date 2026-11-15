#!/usr/bin/env python3
"""swaplink.py BASE_BUILD DONOR_BUILD OUTDIR REGEX [--invert] -- link a libggml-hip.so from BASE_BUILD's HIP objects with the
objects whose path matches REGEX (or, with --invert, all the others) taken from DONOR_BUILD instead. Same link command as
BASE_BUILD's (ggml-hip.dir/link.txt). For finding which objects carry a speed difference between two compilers."""
import os, re, shlex, subprocess, sys
base, donor, outdir, rx = sys.argv[1:5]; invert = '--invert' in sys.argv[5:]
d = base + '/ggml/src/ggml-hip'; dd = donor + '/ggml/src/ggml-hip'
args = shlex.split(open(d + '/CMakeFiles/ggml-hip.dir/link.txt').read())
pat = re.compile(rx); n = tot = 0; new = []
os.makedirs(outdir, exist_ok=True)
i = 0
while i < len(args):
    a = args[i]
    if a == '-o':
        new += ['-o', outdir + '/libggml-hip.so.1.0.0']; i += 2; continue
    if a.endswith('.cu.o'):
        tot += 1
        if bool(pat.search(a)) != invert:
            p = dd + '/' + a
            assert os.path.exists(p), p
            new.append(p); n += 1; i += 1; continue
    new.append(a); i += 1
r = subprocess.run(new, cwd=d, capture_output=True, text=True)
if r.returncode:
    print("LINK FAILED:", r.stderr.strip().splitlines()[-1][:300]); sys.exit(1)
for l in ('libggml-hip.so.1', 'libggml-hip.so'):
    try: os.remove(outdir + '/' + l)
    except OSError: pass
    os.symlink('libggml-hip.so.1.0.0', outdir + '/' + l)
print("%s: %d of %d HIP objects from %s" % (outdir, n, tot, donor))
