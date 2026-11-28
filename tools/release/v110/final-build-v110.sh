#!/bin/bash
# final-build-v110.sh -- the builds of the final master-v110 (the commit at the head of /root/wt-master-v110, which
# must be clean) and the static checks that need no GPU. HOST-CPU HEAVY: refuses beside a GPU job, one build at a time.
#  1. /root/build-final-v110       ROCm 10.0 compiler and tree defaults, with tests: the final suite's build and the
#                                  acceptance's source arm (s100)
#  2. /root/build-final-v110-101   ROCm 10.1 compiler, with tests: the test binary for tbo-pkg-v110.sh, the tile checks
#  3. the package: clean detached worktree /root/wt-deb-v110 at the same commit, ROCMSTREAM=10.1 build-mx-llama-deb.sh
#  4. static, on 2 and on the package's build tree (/root/mx-llama-deb/build):
#     - the LLVM 24 rule (-amdgpu-vgpr-threshold-percent=88) is on exactly ONE compile command, the head-256 tile
#       instance, and on none in build 1 (clang 23 does not know the option)
#     - codegen/tilescan.py on the two tile objects: no kernel with a reload inside a loop (the 88 row of
#       tile/tilescan-thresholds-v110.txt)
#     - codegen/maskcmp.py: the package's GPU code against build 2's (same commit, same compiler, another directory):
#       expected "only immediates differ" everywhere; a function with other instructions is listed
# Every line of the result is in final-build.progress; nothing here decides anything.
set -u
W=/root/rebase-v110; G=/root/exabit-llama.cpp; SRC=/root/wt-master-v110; P=$W/final-build.progress
C=/root/rocm101-migration-20261114/codegen; SUF=fattn-tile-instance-f16-f16-dkq256-dv256.cu
B0=/root/build-final-v110; B1=/root/build-final-v110-101; WD=/root/wt-deb-v110; DB=/root/mx-llama-deb/build
log(){ echo "$(date -Is) [final-build] $*" | tee -a $P; }
for p in llama-server llama-batched-b llama-perplexit test-backend-op test-llama-cont llama-cli cmake; do
  pgrep -x $p >/dev/null && { log "FATAL: $p is running (no builds beside GPU jobs, one build at a time)"; exit 1; }
done
busy=$(cat /sys/class/drm/card[0-9]/device/gpu_busy_percent 2>/dev/null | sort -n | tail -1)
[ "${busy:-0}" -lt 20 ] || { log "FATAL: a die is $busy % busy"; exit 1; }
[ -z "$(git -C $SRC status --porcelain --untracked-files=no)" ] || { log "FATAL: $SRC has uncommitted changes"; exit 1; }
H=$(git -C $SRC rev-parse HEAD)
[ "$H" = "$(git -C $G rev-parse master-v110)" ] || { log "FATAL: $SRC is not at master-v110"; exit 1; }
log "commit $H ($(git -C $SRC log -1 --format=%s | cut -c1-80)); $(git -C $SRC rev-list --count merge-v1.1.0..HEAD) commits over merge-v1.1.0"
build(){ # build DIR ROCMROOT LOGNAME
  if UIVER=b12774 bash $W/build-rocm.sh $SRC $1 $W/$3.log $2 > $W/$3.out 2>&1; then
    echo $H > $1/.src-commit; : > $1/.src-note
    log "built $1 by $2: $(grep 'BUILD rc' $W/$3.out | cut -c1-70); RCCL $(grep -c '^GGML_HIP_RCCL:BOOL=ON' $1/CMakeCache.txt), max-ilp $(grep -c 'amdgpu-sched-strategy=max-ilp' $1/CMakeCache.txt), tests $(ls $1/bin/test-backend-ops $1/bin/test-llama-context $1/bin/test-llama-archs 2>/dev/null | wc -l) of 3"
  else
    log "BUILD FAILED in $1: $(grep -m2 ' error: \|CMake Error' $W/$3.log | tr '\n' ' ' | cut -c1-300)"; log "=== FINAL-BUILD DONE rc=1 ==="; exit 1
  fi
}
rule(){ # rule BUILDDIR -> "<commands with the 88 rule> <of them the tile instance> <commands with any threshold option>"
  python3 -I - $1/compile_commands.json $SUF <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); suf = sys.argv[2]
c = [e for e in d if 'amdgpu-vgpr-threshold-percent=88' in e.get('command', '')]
print(len(c), sum(1 for e in c if e['file'].endswith(suf)), sum(1 for e in d if 'amdgpu-vgpr-threshold-percent' in e.get('command', '')))
PY
}
build $B0 /opt/rocm/core-10.0 build-final-v110
build $B1 /opt/rocm/core-10.1 build-final-v110-101
log "tile rule (commands with =88 / of them the tile instance / commands with any threshold): ROCm 10.0 build $(rule $B0); ROCm 10.1 build $(rule $B1)"
# the package
if [ -d $WD ]; then
  [ -z "$(git -C $WD status --porcelain)" ] || { log "FATAL: $WD exists and is not clean"; exit 1; }
  git -C $WD checkout -q --detach $H || { log "FATAL: checkout of $H in $WD"; exit 1; }
else
  git -C $G worktree add -q --detach $WD $H || { log "FATAL: worktree $WD"; exit 1; }
fi
before=$(cat /root/mx-llama-deb/last-deb.txt 2>/dev/null)
if ROCMSTREAM=10.1 bash /root/mx-llama-deb/build-mx-llama-deb.sh $WD > $W/deb-build-v110.out 2>&1; then
  DEB=$(cat /root/mx-llama-deb/last-deb.txt)
  log "package built: $(basename "$DEB") ($(dpkg-deb -f "$DEB" Version)), source $(dpkg-deb --fsys-tarfile "$DEB" | tar -xO ./usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT | cut -c1-9), $(stat -c %s "$DEB") bytes; last-deb.txt before: $(basename "${before:-none}")"
  log "package Depends: $(dpkg-deb -f "$DEB" Depends | cut -c1-400)"
  log "tile rule in the package build: $(rule $DB)"
else
  log "PACKAGE BUILD FAILED: $(tail -5 $W/deb-build-v110.out | tr '\n' ' ' | cut -c1-400)"; log "=== FINAL-BUILD DONE rc=1 ==="; exit 1
fi
# static: the tile objects and the GPU code
mkdir -p $W/final-static; cd $W/final-static || exit 1
O1=$(find $B1 -name "$SUF.o" | head -1); OD=$(find $DB -name "$SUF.o" | head -1)
{ echo "rows: ROCm 10.1 source build, package build (both $H, threshold 88)"; TMPDIR=$W/final-static python3 $C/tilescan.py "$O1" "$OD"; } > tilescan-final-v110.txt 2>&1
log "tilescan: kernels with a reload inside a loop: source build $(sed -n 3p tilescan-final-v110.txt | tr -s ' ' '\n' | grep -c '^[1-9][0-9]*/'), package build $(sed -n 4p tilescan-final-v110.txt | tr -s ' ' '\n' | grep -c '^[1-9][0-9]*/') of $(sed -n 2p tilescan-final-v110.txt | wc -w) columns minus one (final-static/tilescan-final-v110.txt)"
TMPDIR=$W/final-static python3 -I $C/maskcmp.py $DB $B1 > maskcmp-deb-vs-101.txt 2>&1
log "maskcmp package against the ROCm 10.1 source build: $(tail -4 maskcmp-deb-vs-101.txt | tr '\n' ' ' | cut -c1-400)"
log "=== FINAL-BUILD DONE rc=0 ==="
