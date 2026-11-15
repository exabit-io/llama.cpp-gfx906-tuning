#!/bin/bash
# flag-builds.sh -- the host-compiler variants of mx-llama.cpp master df2c6e5c5 for flag-screen.sh. Every variant is the
# reference configuration (build-rocm.sh, ROCm 10.0 toolchain for the GPU code, GGML_NATIVE=ON) with only the host C/C++
# flags changed; the GPU library is not taken from these builds. Existing: /root/build-master-v100r (no host flags) and
# /root/build-master-v100rn (-march=native, the deb's flags).
#   skx         -march=skylake-avx512
#   v4          -march=x86-64-v4
#   tintel      -march=native -mtune=intel
#   novec       -march=native -fno-tree-vectorize            (diagnostic: the ISA without auto-vectorized code)
#   o2          -O2 instead of CMake's -O3
#   nosemi      -fno-semantic-interposition, shared libraries linked -Bsymbolic-functions
#   lto         -flto=auto
#   ltonosemi   both
#   pgo         ltonosemi + profile-guided: built with -fprofile-generate, trained on ONE run of the screen's own cell
#               (4 x 64K, so a ceiling, not a recipe), rebuilt in the same directory with -fprofile-use
set -u
W=/root/rocm101-migration-20261114; SRC=/root/wt-master-v100; RC=/opt/rocm/core-10.0; M=/root/models/Qwen3.8-27B-Q8_0.gguf
P=$W/flag-builds.progress
log(){ echo "$(date -Is) [flagb] $*" | tee -a $P; }
[ "$(git -C $SRC rev-parse --short HEAD)" = df2c6e5c5 ] && [ -z "$(git -C $SRC status --porcelain --untracked-files=no)" ] || { log "FATAL: source is not a clean df2c6e5c5"; exit 1; }
AR="-DCMAKE_AR=/usr/bin/gcc-ar -DCMAKE_RANLIB=/usr/bin/gcc-ranlib"
one(){ # one NAME  (HOSTFLAGS/RELFLAGS/SHLINK/EXTRA from the caller's environment)
  local n=$1 b=/root/build-flag-$1 rc
  bash $W/build-rocm.sh $SRC $b $W/build-flag-$n.log $RC > /dev/null 2>&1; rc=$?
  log "$n rc=$rc: $(tail -3 $W/build-flag-$n.log | head -1 | cut -c1-80); llama-context.cpp compiled with: $(python3 -I -c '
import json,sys
for e in json.load(open(sys.argv[1])):
    if e["file"].endswith("/src/llama-context.cpp"):
        c = e["command"].split()
        print(" ".join(a for a in c if a.startswith(("-march", "-mtune", "-O", "-flto", "-fno-sem", "-fno-tree", "-fprofile"))))
' $b/compile_commands.json); libllama ymm $(objdump -d $b/bin/libllama.so.1 | grep -c '%ymm') zmm $(objdump -d $b/bin/libllama.so.1 | grep -c '%zmm')"
  return $rc
}
HOSTFLAGS="-march=skylake-avx512" one skx || exit 1
HOSTFLAGS="-march=x86-64-v4" one v4 || exit 1
HOSTFLAGS="-march=native -mtune=intel" one tintel || exit 1
HOSTFLAGS="-march=native -fno-tree-vectorize" one novec || exit 1
RELFLAGS="-O2 -DNDEBUG" one o2 || exit 1
HOSTFLAGS="-fno-semantic-interposition" SHLINK="-Wl,-Bsymbolic-functions" one nosemi || exit 1
HOSTFLAGS="-flto=auto" EXTRA="$AR" one lto || exit 1
HOSTFLAGS="-flto=auto -fno-semantic-interposition" SHLINK="-Wl,-Bsymbolic-functions" EXTRA="$AR" one ltonosemi || exit 1
# profile-guided: instrumented build, one training run of the cell on the ROCm 10.1 runtime, rebuild with the profile
HOSTFLAGS="-flto=auto -fno-semantic-interposition -fprofile-generate -fprofile-update=atomic" SHLINK="-Wl,-Bsymbolic-functions" EXTRA="$AR" one pgo || exit 1
B=/root/build-flag-pgo
log "pgo training run: 4 x 64K on the instrumented build"
env LD_LIBRARY_PATH=$B/bin:/opt/rocm/core-10.1/lib AMD_COMGR_CACHE_DIR=/root/.cache/comgr-flag-pgo-train \
  GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1 timeout 3600 \
  $B/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 \
  -c $(( 4*(65536+1280) )) -npp 65536 -ntg 1024 -npl 4 > $W/flag-pgo-train.md 2> $W/flag-pgo-train.log
rc=$?
ng=$(find $B -name '*.gcda' | wc -l)
log "pgo training rc=$rc, $ng profile files; $(grep -E '^\|' $W/flag-pgo-train.md | tail -1 | cut -c1-120)"
[ $rc = 0 ] && [ "$ng" -gt 50 ] || { log "FATAL: no usable profile"; exit 1; }
mkdir -p $W/flag-pgo-instrumented && cp -a $B/bin/libllama.so.1.0.0 $W/flag-pgo-instrumented/ 2>/dev/null
HOSTFLAGS="-flto=auto -fno-semantic-interposition -fprofile-use -fprofile-correction -Wno-missing-profile" SHLINK="-Wl,-Bsymbolic-functions" EXTRA="$AR" one pgo || exit 1
log "profile warnings in the pgo build: $(grep -c 'profile' $W/build-flag-pgo.log)"
log "=== FLAG BUILDS DONE ==="
