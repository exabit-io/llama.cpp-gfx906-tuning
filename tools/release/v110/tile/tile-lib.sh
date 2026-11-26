#!/bin/bash
# tile-lib.sh TAG OBJECT -- link /root/rebase-v110/tile/lib-TAG/libggml-hip.so from the objects of the ROCm 10.1 build of
# master-v110 (/root/build-master-v110-101, its own link command) with ONLY the head-256 tile object taken from OBJECT
# (tile/thr/try-tNN.o of tile-thr-v110.sh, or the clang 23 object of /root/build-master-v110). For timing one object.
set -eu
TAG=$1; OBJ=$(readlink -f $2); B=/root/build-master-v110-101; D=$B/ggml/src/ggml-hip; O=/root/rebase-v110/tile/lib-$TAG
mkdir -p $O
python3 -I - "$D" "$OBJ" "$O" <<'PY'
import os, shlex, subprocess, sys
d, obj, out = sys.argv[1:4]
args = shlex.split(open(d + '/CMakeFiles/ggml-hip.dir/link.txt').read())
new = []; n = 0; i = 0
while i < len(args):
    a = args[i]
    if a == '-o':
        new += ['-o', out + '/libggml-hip.so.1.0.0']; i += 2; continue
    if a.endswith('fattn-tile-instance-f16-f16-dkq256-dv256.cu.o'):
        new.append(obj); n += 1; i += 1; continue
    new.append(a); i += 1
assert n == 1, n
subprocess.run(new, cwd=d, check=True)
PY
ln -sf libggml-hip.so.1.0.0 $O/libggml-hip.so.1; ln -sf libggml-hip.so.1 $O/libggml-hip.so
echo "lib-$TAG: $(stat -c %s $O/libggml-hip.so.1.0.0) bytes, tile object $(sha256sum < $OBJ | cut -c1-12)"
