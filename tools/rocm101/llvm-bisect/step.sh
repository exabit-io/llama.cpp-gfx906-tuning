#!/bin/bash
# step.sh [LABEL] -- build llc at the checked-out commit of /root/llvm-bisect/src and compile clang 23's optimized bitcode
# of the head-256 tile instance with it, as the tree does (-O3, gfx906, max-ilp scheduler). Reads the 16x2 kernel's
# largest loop with codegen/loopstat.py. Exit status for 'git bisect run':
#   0   good: at most 1 scratch reload (buffer_load) inside that loop (ROCm 10.0's compiler: 0)
#   1   bad: 5 or more (ROCm 10.1's compiler: 10)
#   125 skip: the commit does not build, llc fails on the bitcode, or the count is 2-4
# One line per call is appended to steps.log. Heavy (56 jobs): never while a timing run is in progress.
# Environment: LLCFLAGS (further llc options), SCHED (replaces '-amdgpu-sched-strategy=max-ilp'; empty = the default
# scheduler), BC (another bitcode file).
set -u
T=/root/llvm-bisect; W=/root/rocm101-migration-20261114/llvm-bisect; BC=${BC:-/root/rocm101-migration-20261114/codegen/irx-x23on23.bc}
pgrep -x llama-batched-b >/dev/null && { echo "FATAL: a timing run is in progress"; exit 255; }
cd $T/src || exit 255
H=$(git rev-parse --short=12 HEAD); L=${1:-$H}
say(){ echo "$(date -Is) $H $(git log -1 --format=%cd --date=short) $*" | tee -a $W/steps.log; }
conf(){ cmake -G Ninja -S $T/src/llvm -B $T/build -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=AMDGPU -DLLVM_ENABLE_PROJECTS= \
  -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_APPEND_VC_REV=OFF -DLLVM_CCACHE_BUILD=ON > $W/cmake-$L.log 2>&1; }
build(){ nice -n 5 ninja -C $T/build -j 56 llc > $W/build-$L.log 2>&1; }
t0=$(date +%s)
[ -f $T/build/build.ninja ] || conf
if ! build; then
  rm -rf $T/build
  if ! conf || ! build; then say "SKIP: does not build: $(grep -m1 -E 'error:|CMake Error' $W/build-$L.log $W/cmake-$L.log | cut -c1-160)"; exit 125; fi
fi
t1=$(date +%s)
O=$W/out-$L.o
if ! timeout 3600 $T/build/bin/llc -O3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -relocation-model=pic ${SCHED--amdgpu-sched-strategy=max-ilp} ${LLCFLAGS:-} \
     -filetype=obj -o $O $BC > $W/llc-$L.log 2>&1; then
  say "SKIP: llc failed: $(tail -1 $W/llc-$L.log | cut -c1-160)"; exit 125
fi
t2=$(date +%s)
row=$(python3 -I /root/rocm101-migration-20261114/codegen/loopstat.py $O | tail -1)
# columns after the '|': hot loop insns, gload, reload, spill, waits, lds, v_mov
r=$(echo "$row" | awk -F'|' '{split($2, a, " "); print a[3]}')
case $r in ''|*[!0-9]*) say "SKIP: no reading: $(echo "$row" | cut -c1-160)"; exit 125;; esac
v=good; rc=0
if [ "$r" -ge 5 ]; then v=bad; rc=1; elif [ "$r" -ge 2 ]; then v=ambiguous; rc=125; fi
say "$v: $r reloads in the hot loop [$(echo "$row" | awk -F'|' '{print $2}' | tr -s ' ')] build $((t1 - t0))s llc $((t2 - t1))s ${1:+label $1} ${LLCFLAGS:+flags $LLCFLAGS} ${SCHED+sched '$SCHED'}"
exit $rc
