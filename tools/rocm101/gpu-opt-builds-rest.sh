#!/bin/bash
# gpu-opt-builds-rest.sh -- the 18 survey variants that gpu-opt-builds.sh had not built when it was stopped for the tile
# timing run (tid, o2, ifcvt exist), three builds at a time (the tail of one build is a few long template compiles), and
# after them the candidate for the ROCm 10.1 compiler: /root/wt-master-t90 (df2c6e5c5 + the CMake rule that compiles the
# head-256 tile instance with -amdgpu-vgpr-threshold-percent=90 on LLVM >= 24) built by /opt/rocm/core-10.1 into
# /root/build-v101-t90. Its GPU code must differ from /root/build-master-v101 in exactly that one object.
# Heavy: never while a timing run is in progress.
set -u
W=/root/rocm101-migration-20261114; P=$W/gpu-opt-builds.progress
log(){ echo "$(date -Is) [gpuopt] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
cd $W || exit 1
rm -rf /root/build-gpuopt-noslp    # left half-built when the first round was stopped
bash $W/gpu-opt-builds.sh noslp unrollhi inlthr sgl0 dividx schmem > $W/gpu-opt-builds-1.out 2>&1 &
p1=$!
bash $W/gpu-opt-builds.sh novec unrolllo relax lsv0 spillspd schiilp > $W/gpu-opt-builds-2.out 2>&1 &
p2=$!
bash $W/gpu-opt-builds.sh unrollf calls0 trk pa64 schdef schiocc > $W/gpu-opt-builds-3.out 2>&1 &
p3=$!
wait $p1; wait $p2; wait $p3
B=/root/build-v101-t90
if bash $W/build-rocm.sh /root/wt-master-t90 $B $W/build-v101-t90.log /opt/rocm/core-10.1 > /dev/null 2>&1; then
  log "t90: $(grep -m1 '^BUILD rc=' $W/build-v101-t90.log | cut -c1-60); tile instance compiled with threshold: $(grep -c 'amdgpu-vgpr-threshold-percent=90' $B/compile_commands.json) command(s)"
  log "t90 against the plain ROCm 10.1 build: $(python3 -I $W/codegen/devcmp.py /root/build-master-v101 $B | cut -c1-400)"
else
  log "t90 BUILD FAILED: $(grep -m1 -E ' error: |CMake Error' $W/build-v101-t90.log | cut -c1-200)"
fi
log "=== REST OF THE BUILDS DONE ==="
