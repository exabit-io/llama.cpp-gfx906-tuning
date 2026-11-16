#!/bin/bash
# gpu-opt24-builds.sh [NAME ...] -- the GPU compile-option survey again, for the compiler the package is now built with:
# ROCm 10.1's clang 24 on master 290fc642a (/root/wt-master-t90, the tree with the LLVM 24 tile rule). The first survey
# (gpu-opt-builds.sh, report section 5.4) used ROCm 10.0's clang 23. One change per variant against the reference
# /root/build-v101-t90 (the GPU code of the -r1 package), each a whole build into /root/build-gpuopt24-NAME.
#
# The variants are those of gpu-opt-builds.sh (definitions there), without one and with four more:
#   tid      NOT built: with a full target ID the tree applies neither max-ilp nor the tile rule (both test for the
#            architecture name 'gfx906'), so the variant would differ from the reference in three things; it was level
#            under clang 23 and a build for 'sramecc-' dies cannot ship to MI50s anyway.
#   schiilp  iterative-ilp on every source: ROCm 10.0's clang 23 crashed on it; built here to see whether clang 24 does.
#   v80      -amdgpu-vgpr-threshold-percent=80 on every source (the option is new in LLVM 24; 90 on every source is the
#            existing /root/build-v101-v90). Built from /root/wt-master-v100 (df2c6e5c5, the tree WITHOUT the rule),
#            because the option may be given only once per compile: the tile instance gets 80 like every other source.
#   urt1 urt0  -amdgpu-unroll-runtime-default=1 / =0 (new in LLVM 24); the one that equals the compiler's default
#            produces the reference's code and is not timed.
#   nolicm   -disable-machine-licm on every source (added 2026-11-16 after the bisect, report section 4.2: ROCm 10.1's
#            compiler runs machine LICM without a vector-register limit in every kernel; this arm asks whether any
#            kernel is better off without the pass). Not in the default list; build it by name.
# The scheduler variants (schdef, schmem, schiilp, schiocc) switch GGML_HIP_GFX906_MAX_ILP off, and with it the tile
# rule, which the tree ties to max-ilp.
#
# After each build: the HIP flags as recorded, and how much GPU machine code differs from the reference (devcmp.py:
# objects; fncmp.py: functions). A variant whose GPU code is byte-identical needs no timing.
# Heavy (56 compile jobs per build): never while a timing run is in progress.
set -u
W=/root/rocm101-migration-20261114; RC=/opt/rocm/core-10.1; REF=/root/build-v101-t90
P=$W/gpu-opt24-builds.progress
log(){ echo "$(date -Is) [gpuopt24] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
clean_at(){ [ "$(git -C $1 rev-parse --short=9 HEAD)" = "$2" ] && [ -z "$(git -C $1 status --porcelain --untracked-files=no)" ]; }
clean_at /root/wt-master-t90 290fc642a || { log "FATAL: /root/wt-master-t90 is not a clean 290fc642a"; exit 1; }
clean_at /root/wt-master-v100 df2c6e5c5 || { log "FATAL: /root/wt-master-v100 is not a clean df2c6e5c5"; exit 1; }
[ -f $REF/bin/libggml-hip.so ] || { log "FATAL: no reference build $REF"; exit 1; }
declare -A HF=() HR=() EX=() SR=()
HR[o2]='-O2 -DNDEBUG'
HF[ifcvt]='-mllvm -amdgpu-early-ifcvt=1'
HF[noslp]='-fno-slp-vectorize'
HF[novec]='-fno-vectorize -fno-slp-vectorize'
HF[unrollf]='-funroll-loops'
HF[unrollhi]='-mllvm -unroll-threshold=3000'
HF[unrolllo]='-mllvm -unroll-threshold=300'
HF[calls0]='-mllvm -amdgpu-function-calls=0'
HF[inlthr]='-mllvm -inline-threshold=1000'
HF[relax]='-mllvm -amdgpu-schedule-relaxed-occupancy=1'
HF[trk]='-mllvm -amdgpu-use-amdgpu-trackers=1'
HF[sgl0]='-mllvm -amdgpu-scalarize-global-loads=0'
HF[lsv0]='-mllvm -amdgpu-load-store-vectorizer=0'
HF[pa64]='-mllvm -amdgpu-promote-alloca-to-vector-max-regs=64 -mllvm -amdgpu-promote-alloca-to-vector-vgpr-ratio=2'
HF[dividx]='-mllvm -amdgpu-use-divergent-register-indexing=1'
HF[spillspd]='-mllvm -split-spill-mode=speed'
EX[schdef]='-DGGML_HIP_GFX906_MAX_ILP=OFF'
EX[schmem]='-DGGML_HIP_GFX906_MAX_ILP=OFF'; HF[schmem]='-mllvm -amdgpu-sched-strategy=max-memory-clause'
EX[schiilp]='-DGGML_HIP_GFX906_MAX_ILP=OFF'; HF[schiilp]='-mllvm -amdgpu-sched-strategy=iterative-ilp'
EX[schiocc]='-DGGML_HIP_GFX906_MAX_ILP=OFF'; HF[schiocc]='-mllvm -amdgpu-sched-strategy=iterative-maxocc'
HF[v80]='-mllvm -amdgpu-vgpr-threshold-percent=80'; SR[v80]=/root/wt-master-v100
HF[urt1]='-mllvm -amdgpu-unroll-runtime-default=1'
HF[urt0]='-mllvm -amdgpu-unroll-runtime-default=0'
HF[nolicm]='-mllvm -disable-machine-licm'
ALL="o2 ifcvt noslp novec unrollf unrollhi unrolllo calls0 inlthr relax trk sgl0 lsv0 pa64 dividx spillspd schdef schmem schiilp schiocc v80 urt1 urt0"
one(){ # one NAME
  local n=$1 b=/root/build-gpuopt24-$1 rc
  HIPFLAGS=${HF[$n]:-} HIPREL=${HR[$n]:-} EXTRA=${EX[$n]:-} \
    bash $W/build-rocm.sh ${SR[$n]:-/root/wt-master-t90} $b $W/build-gpuopt24-$n.log $RC > /dev/null 2>&1; rc=$?
  if [ $rc != 0 ]; then
    log "$n BUILD FAILED rc=$rc: $(grep -m1 -E ' error: |CMake Error|PLEASE submit a bug report|Stack dump' $W/build-gpuopt24-$n.log | cut -c1-200)"; return 1
  fi
  log "$n: $(grep -m1 '^BUILD rc=' $W/build-gpuopt24-$n.log | cut -c1-60); tile instance compiled with: $(python3 -I -c '
import json, shlex, sys
for e in json.load(open(sys.argv[1])):
    if e["file"].endswith("/fattn-tile-instance-dkq256-dv256.cu"):
        c = shlex.split(e["command"]); keep = []; skip = False
        for i, a in enumerate(c[1:]):
            if skip: skip = False; continue
            if a in ("-o", "-c", "-x"): skip = a != "-c"; continue
            if a.startswith(("-D", "-I", "-isystem", "/")): continue
            keep.append(a)
        print(" ".join(keep))' $b/compile_commands.json)"
  log "$n: $(python3 -I $W/codegen/devcmp.py $REF $b | cut -c1-400)"
  log "$n: $(python3 -I $W/codegen/fncmp.py $REF $b | cut -c1-300)"
}
for n in ${*:-$ALL}; do
  [ -n "${HF[$n]:-}${HR[$n]:-}${EX[$n]:-}" ] || { log "unknown variant $n"; continue; }
  one $n
done
log "=== GPU-OPT24 BUILDS DONE ($*) ==="
