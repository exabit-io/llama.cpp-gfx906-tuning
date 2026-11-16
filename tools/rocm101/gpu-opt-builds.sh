#!/bin/bash
# gpu-opt-builds.sh [NAME ...] -- variants of mx-llama.cpp master df2c6e5c5 for the GPU compile-option survey (the lead,
# 2026-11-15: "survey GPU-side compile options"). Every variant is the reference configuration (build-rocm.sh, ROCm 10.0
# toolchain = the compiler the package ships with, no host flags) with ONE change to how the HIP sources are compiled.
# Reference: /root/build-master-v100r, HIP flags '-mllvm -amdgpu-sched-strategy=max-ilp -O3', architecture 'gfx906'.
# Only options that leave floating-point semantics alone are here (no fast-math, no contraction or denormal changes),
# so a variant must reproduce the reference's outputs; that is checked on whatever the timing screen keeps.
#
#   tid       target ID gfx906:sramecc-:xnack- (what the dies report) instead of generic gfx906: lets the compiler use
#             16-bit loads that keep the other half of a register (off while SRAM ECC may be on) and 4 more SGPRs
#   o2        -O2 instead of -O3
#   ifcvt     -amdgpu-early-ifcvt=1: branches to selects before register allocation
#   noslp     -fno-slp-vectorize                     novec    -fno-vectorize -fno-slp-vectorize
#   unrollf   -funroll-loops                         unrollhi -unroll-threshold=3000     unrolllo -unroll-threshold=300
#   calls0    -amdgpu-function-calls=0: every device function inlined
#   inlthr    -inline-threshold=1000 (the device default is 225 x 11 for AMDGPU; forceinline functions are unaffected)
#   relax     -amdgpu-schedule-relaxed-occupancy=1   trk      -amdgpu-use-amdgpu-trackers=1
#   sgl0      -amdgpu-scalarize-global-loads=0       lsv0     -amdgpu-load-store-vectorizer=0
#   pa64      -amdgpu-promote-alloca-to-vector-max-regs=64 -amdgpu-promote-alloca-to-vector-vgpr-ratio=2
#   dividx    -amdgpu-use-divergent-register-indexing=1
#   spillspd  -split-spill-mode=speed
#   schdef    the compiler's default scheduler (GGML_HIP_GFX906_MAX_ILP=OFF): the tree's own comment puts max-ilp at
#             +2.5 % decode over it, so this arm is the screen's positive control
#   schmem    -amdgpu-sched-strategy=max-memory-clause   schiilp  iterative-ilp   schiocc  iterative-maxocc
#
# After each build: the HIP flags as recorded, and how much GPU machine code differs from the reference (devcmp.py:
# objects; fncmp.py: functions). A variant whose GPU code is byte-identical needs no timing.
# Heavy (56 compile jobs): never while a timing run is in progress.
set -u
W=/root/rocm101-migration-20261114; SRC=/root/wt-master-v100; RC=/opt/rocm/core-10.0; REF=/root/build-master-v100r
P=$W/gpu-opt-builds.progress
log(){ echo "$(date -Is) [gpuopt] $*" | tee -a $P; }
[ "$(git -C $SRC rev-parse --short HEAD)" = df2c6e5c5 ] && [ -z "$(git -C $SRC status --porcelain --untracked-files=no)" ] || { log "FATAL: source is not a clean df2c6e5c5"; exit 1; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
ILP="-mllvm -amdgpu-sched-strategy=max-ilp"
declare -A HF=() HR=() HA=() EX=()
HA[tid]='gfx906:sramecc-:xnack-'; HF[tid]=$ILP
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
ALL="tid o2 ifcvt noslp novec unrollf unrollhi unrolllo calls0 inlthr relax trk sgl0 lsv0 pa64 dividx spillspd schdef schmem schiilp schiocc"
one(){ # one NAME
  local n=$1 b=/root/build-gpuopt-$1 rc
  HIPFLAGS=${HF[$n]:-} HIPREL=${HR[$n]:-} HIPARCH=${HA[$n]:-} EXTRA=${EX[$n]:-} \
    bash $W/build-rocm.sh $SRC $b $W/build-gpuopt-$n.log $RC > /dev/null 2>&1; rc=$?
  if [ $rc != 0 ]; then
    log "$n BUILD FAILED rc=$rc: $(grep -m1 -E ' error: |CMake Error' $W/build-gpuopt-$n.log | cut -c1-200)"; return 1
  fi
  log "$n: $(grep -m1 '^BUILD rc=' $W/build-gpuopt-$n.log | cut -c1-60); mmvq.cu compiled with: $(python3 -I -c '
import json, shlex, sys
for e in json.load(open(sys.argv[1])):
    if e["file"].endswith("/ggml-cuda/mmvq.cu"):
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
  [ -n "${HF[$n]:-}${HR[$n]:-}${HA[$n]:-}${EX[$n]:-}" ] || { log "unknown variant $n"; continue; }
  one $n
done
log "=== GPU-OPT BUILDS DONE ==="
