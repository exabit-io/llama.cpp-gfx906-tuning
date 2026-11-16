#!/bin/bash
# gpu-opt24-builds-all.sh -- the 23 variants of gpu-opt24-builds.sh, three builds at a time (the tail of one build is a
# few long template compiles), and after them the ROCm 10.0-stream package of master 290fc642a without the global
# -march=native (build-mx-llama-deb.sh with ROCMSTREAM=10.0: ROCm 10.0's clang 23 and runtime).
# Heavy: never while a timing run is in progress.
set -u
W=/root/rocm101-migration-20261114; P=$W/gpu-opt24-builds.progress
log(){ echo "$(date -Is) [gpuopt24] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
cd $W || exit 1
bash $W/gpu-opt24-builds.sh o2 unrollf calls0 trk dividx schmem v80 urt0 > $W/gpu-opt24-builds-1.out 2>&1 &
p1=$!
bash $W/gpu-opt24-builds.sh ifcvt novec unrollhi inlthr sgl0 spillspd schiilp urt1 > $W/gpu-opt24-builds-2.out 2>&1 &
p2=$!
bash $W/gpu-opt24-builds.sh noslp unrolllo relax lsv0 pa64 schdef schiocc > $W/gpu-opt24-builds-3.out 2>&1 &
p3=$!
wait $p1; wait $p2; wait $p3
log "=== ALL THREE STREAMS DONE: $(ls -d /root/build-gpuopt24-*/bin/libggml-hip.so 2>/dev/null | wc -l) of 23 libraries ==="
cp -p /root/mx-llama-deb/last-deb.txt /root/mx-llama-deb/last-deb.txt.pre-rocm100-r1 2>/dev/null
if ROCMSTREAM=10.0 COMPILERSTREAM=10.0 bash /root/mx-llama-deb/build-mx-llama-deb.sh /root/wt-master-t90 > $W/deb100-build.out 2>&1; then
  log "10.0-stream package built: $(cat /root/mx-llama-deb/last-deb.txt 2>/dev/null)"
else
  log "10.0-stream package BUILD FAILED: $(tail -3 $W/deb100-build.out | tr '\n' ' ' | cut -c1-300)"
fi
log "=== GPU-OPT24 BUILDS-ALL DONE ==="
