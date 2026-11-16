#!/bin/bash
# drv-test.sh [N] -- is the ROCm 10.1 compiler's prefill cost (head-256 flash-attention tile kernel; 1 x 255K prefill
# -4.1 %, all of it in the one object fattn-tile-instance-dkq256-dv256) the compiler's code generation, or the way its
# driver compiles device code? clang 24's HIP driver defaults to the "new offload driver": device code is emitted as
# bitcode (-flto=full) and compiled at link time by the full-LTO pipeline. clang 23's default compiled it directly
# (cc1 -emit-obj). Both compilers can do both, which gives a 2 x 2:
#   c23     clang 23, direct (as the reference build)        c23lto   clang 23 --offload-new-driver -foffload-lto=full
#   c24     clang 24, full LTO (as the 10.1 build)           c24old   clang 24 --no-offload-new-driver (direct, old driver)
#   c24nol  clang 24 -fno-offload-lto (direct, new driver)   c24thin  clang 24 -foffload-lto=thin
# Every arm is the ROCm 10.1 build (/root/build-master-v101) on the 10.1 runtime with libggml-hip.so relinked from that
# build's objects and ONLY the one tile object replaced (codegen/drvexp.py); c24 is relinked the same way from an
# unchanged recompile, so the link itself is controlled.
# Cell: 1 x 131072 prefill, -ntg 256 (screen.sh), N blocks (default 2), 4 min per cell.
#
# READING, WRITTEN BEFORE THE FIRST CELL. Statistic: prefill tok/s. The c23/c24 gap was 15 tok/s (2.6 %) with single
# samples within 2 tok/s of each other. (1) If c23 and c24 do not differ by more than 8 tok/s in every block, the gap is
# not reproduced and nothing is read. (2) An arm is "fast" when all its samples are within 4 tok/s of c23's mean, "slow"
# when all are within 4 tok/s of c24's mean, otherwise "between". (3) The driver's LTO pipeline is the cause if c24old or
# c24nol is fast AND c23lto is slow; the compiler's code generation is the cause if both clang 24 direct arms are slow
# and c23lto is fast; anything else is mixed and is reported as such.
set -u
W=/root/rocm101-migration-20261114; C=$W/codegen; N=${1:-2}; TAG=drv128k
P=$W/drv-test.progress
log(){ echo "$(date -Is) [drv] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
mkdir -p /root/swap
cd $C || exit 1
BP=()
build(){ # build NAME STREAM [FLAGS...]  (in the background; output to drv-NAME.out)
  python3 -I $C/drvexp.py "$@" > $C/drv-$1.out 2>&1 &
  BP+=($!)
}
build c23 100; build c23lto 100 --offload-new-driver -foffload-lto=full
build c24 101; build c24old 101 --no-offload-new-driver; build c24nol 101 -fno-offload-lto; build c24thin 101 -foffload-lto=thin
for p in "${BP[@]}"; do wait "$p"; done
ARMS=()
for a in c23 c24 c24old c24nol c23lto c24thin; do
  log "$(tr '\n' ' ' < $C/drv-$a.out | cut -c1-600)"
  if [ -f /root/swap/$a/libggml-hip.so.1.0.0 ]; then ARMS+=("$a=/root/build-master-v101:/opt/rocm/core-10.1:/root/swap/$a/libggml-hip.so"); else log "$a: no library, arm dropped"; fi
done
python3 -I $C/loopstat.py $C/tile-v100r.elf $C/tile-v101.elf $C/drv-c23.elf $C/drv-c24.elf $C/drv-c24old.elf $C/drv-c24nol.elf $C/drv-c23lto.elf $C/drv-c24thin.elf 2>&1 | tee -a $P
for a in c23 c24 c24old c24nol c23lto c24thin; do
  [ -f $C/drv-$a.elf ] && log "GPU code of $a: sha256 of .text $(python3 -I -c '
import hashlib, struct, sys
b = open(sys.argv[1], "rb").read()
shoff, = struct.unpack_from("<Q", b, 40); es, n, sx = struct.unpack_from("<HHH", b, 58)
so, = struct.unpack_from("<Q", b, shoff + sx * es + 24)
for k in range(n):
    nm, = struct.unpack_from("<I", b, shoff + k * es); off, size = struct.unpack_from("<QQ", b, shoff + k * es + 24)
    if b[so + nm:b.find(b"\0", so + nm)] == b".text": print(hashlib.sha256(b[off:off + size]).hexdigest()[:16], size)' $C/drv-$a.elf)"
done
[ "${#ARMS[@]}" -ge 4 ] || { log "FATAL: fewer than four arms built"; exit 1; }
[ "${BUILDONLY:-0}" = 1 ] && { log "build only"; exit 0; }
cd $W && bash $W/screen.sh $TAG 131072 1 256 "$N" "${ARMS[@]}"
log "=== DRV TEST DONE ==="
