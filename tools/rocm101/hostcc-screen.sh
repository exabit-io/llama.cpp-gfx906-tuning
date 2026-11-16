#!/bin/bash
# hostcc-screen.sh -- the host COMPILER and 4 x 64K decode. The flag screen (flag-screen.sh, report section 5.2) found no
# gcc 13 flag set better than CMake Release defaults + GGML_NATIVE and left one question open: what another host compiler
# does. hostcc-builds.sh has the arms.
#
# Method (flag-screen.sh's): the installed package's llama-batched-bench and its GPU library, with the six host libraries
# (libggml-base, libggml, libggml-cpu, libllama, libllama-common, libllama-batched-bench-impl) of one variant build
# preloaded by full path. One source commit (290fc642a), one GPU library, the ROCm 10.1 runtime in every arm; only the
# host compiler differs. GGML_NATIVE=ON in every arm.
#   base      gcc 13.3, OpenMP (/root/build-v101-t90: the -r1 package's configuration)
#   gccnoomp  gcc 13.3, GGML_OPENMP=OFF      c19  clang 19, GGML_OPENMP=OFF      c24  ROCm 10.1's clang 24, GGML_OPENMP=OFF
# Cell: the multi-user design point (Qwen3.8-27B Q8_0, 4 x 64K, -sm tensor, f16 KV, 125 W, RCCL + gated custom AR, -ntg 1024).
#
# READING, WRITTEN BEFORE THE FIRST CELL: a screen, not an acceptance test. n = N blocks (default 3), every arm once per
# block in a seeded random order. Statistic: mean decode per slot; prefill is listed.
# (1) gccnoomp against base says what OpenMP is worth in this cell; they "match" when their ranges overlap.
# (2) A clang arm is "up" when every one of its runs is above every run of BOTH gcc arms, "down" when every one is below
#     every run of both, otherwise "level".
# (3) A clang arm is taken further (a package built with it, then the design-point acceptance) only if it is "up" AND its
#     mean is 0.3 % or more above base AND its prefill mean is not more than 0.25 % below base's: a second host compiler
#     is a build dependency and, with OpenMP, a second OpenMP runtime, and a smaller gain does not pay for that.
#     Anything else is reported as measured and dropped.
set -u
W=/root/rocm101-migration-20261114; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
TL=/root/llama.cpp-benchmarking/tools; CM=$TL/cell-metrics.py; NF=$TL/assert-no-fa-fallback.sh
N=${N:-3}; ARMS=${ARMS:-"base gccnoomp c19 c24"}; TAG=${TAG:-hostcc-screen}
P=$W/$TAG.progress; TSV=$W/$TAG.tsv; DONE=$W/.$TAG-done; O=$W/$TAG; mkdir -p $O
log(){ echo "$(date -Is) [hostcc] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE; echo $$ > $W/$TAG.pid
PK=/opt/mx-llama.cpp/lib; RT=/opt/rocm/core-10.1; EXE=/opt/mx-llama.cpp/bin/llama-batched-bench
HOST="libggml-base.so.1 libggml.so.1 libggml-cpu.so.1 libllama.so.1 libllama-common.so.1 libllama-batched-bench-impl.so"
arm_dir(){ case $1 in base) echo /root/build-v101-t90/bin;; *) echo /root/build-hostcc-$1/bin;; esac; }
want(){ # want ARM -> the seven libraries of the arm by resolved full path, one per line
  local d l; d=$(arm_dir $1)
  for l in $HOST; do readlink -f $d/$l; done
  readlink -f $PK/libggml-hip.so.1
}
preload(){ want "$1" | tr '\n' ':' | sed 's/:$//'; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; touch $DONE; exit 1; }
[ "$(git -C /root/wt-master-t90 rev-parse HEAD)" = "$(cat /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT)" ] \
  || { log "FATAL: the package and the variant builds are not the same commit"; touch $DONE; exit 1; }
dpkg-query -W -f '${Version}' mx-llama-cpp-gfx906 | grep -q 'rocm10\.1\.' || { log "FATAL: the installed package is not the ROCm 10.1 build"; touch $DONE; exit 1; }
for a in $ARMS; do
  d=$(arm_dir $a)
  for l in $HOST; do [ -f $d/$l ] || { log "FATAL: arm $a has no $d/$l"; touch $DONE; exit 1; }; done
  [ "$(grep -m1 -oE 'CMAKE_HOME_DIRECTORY:INTERNAL=.*' ${d%/bin}/CMakeCache.txt | cut -d= -f2)" = /root/wt-master-t90 ] || { log "FATAL: arm $a was not built from /root/wt-master-t90"; touch $DONE; exit 1; }
  got=$(LD_LIBRARY_PATH=$RT/lib LD_PRELOAD=$(preload $a) ldd $EXE 2>/dev/null | awk '{ if ($2 == "=>") print $3; else print $1 }' | grep -E '/lib(ggml|llama)' | xargs -r -n1 readlink -f | sort -u)
  exp=$(want $a | sort -u)
  [ "$got" = "$exp" ] || { log "FATAL: arm $a maps other libraries than intended: $(echo "$got" | tr '\n' ' ')"; touch $DONE; exit 1; }
  nro=$(LD_LIBRARY_PATH=$RT/lib LD_PRELOAD=$(preload $a) ldd $EXE | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | wc -l)
  bad=$(LD_LIBRARY_PATH=$RT/lib LD_PRELOAD=$(preload $a) ldd $EXE | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | grep -vc "^$RT/")
  [ "$bad" = 0 ] && [ "$nro" -gt 5 ] || { log "FATAL: arm $a resolves $bad of $nro ROCm libraries outside $RT"; touch $DONE; exit 1; }
  log "arm $a: host libraries from $d; C++ compiler $(grep -E '^CMAKE_CXX_COMPILER:' ${d%/bin}/CMakeCache.txt | cut -d= -f2), OpenMP $(grep -E '^GGML_OPENMP_ENABLED:' ${d%/bin}/CMakeCache.txt | cut -d= -f2), host flags [$(grep -E '^CMAKE_CXX_FLAGS:' ${d%/bin}/CMakeCache.txt | cut -d= -f2)]"
done
log "n=$N, arms: $ARMS; load $(cut -d' ' -f1-3 /proc/loadavg)"

. $R/gpu-test-env.sh
export GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
BPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${BPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/$TAG-clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/$TAG-smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $W/$TAG-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/$TAG-smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
log "$(state_line)"
cell(){ # cell ARM BLOCK
  local arm=$1 blk=$2 s=4 d=65536
  local out=$O/multi-$arm-$blk.md child got exp
  env LD_LIBRARY_PATH=$RT/lib LD_PRELOAD="$(preload $arm)" AMD_COMGR_CACHE_DIR=/root/.cache/comgr-hostcc timeout 7200 \
    $EXE -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s \
    > $out 2>${out%.md}.log &
  BPID=$!
  sleep 45      # the libraries are mapped long before; the model is still loading
  child=$(pgrep -P $BPID | head -1)
  got=$(awk '$6 ~ /\/lib(ggml|llama)/ {print $6}' /proc/${child:-0}/maps 2>/dev/null | sort -u)
  exp=$(want $arm | sort -u)
  echo "$got" > ${out%.md}.maps
  wait $BPID
  local rc=$?
  BPID=""
  if [ "$got" != "$exp" ]; then
    log "WRONG LIBRARIES in $arm b$blk -- discarded: $(echo "$got" | tr '\n' ' ')"; printf "multi\t%s\t%s\tWRONGLIBS\tWRONGLIBS\n" "$arm" "$blk" >> $TSV; return 0
  fi
  if ! $NF "${out%.md}.log" >/dev/null 2>&1; then
    log "FA FALLBACK in $arm b$blk -- discarded"; printf "multi\t%s\t%s\tFALLBACK\tFALLBACK\n" "$arm" "$blk" >> $TSV; return 0
  fi
  if grep -qiE 'NCCL init failed|probe AllReduce failed' "${out%.md}.log"; then
    log "RCCL NOT UP in $arm b$blk -- discarded"; printf "multi\t%s\t%s\tNORCCL\tNORCCL\n" "$arm" "$blk" >> $TSV; return 0
  fi
  local m
  if ! m=$(python3 -I $CM "$out" $d $s 2>>$P); then
    log "CELL FAILED (rc=$rc): $arm b$blk"; printf "multi\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$blk" >> $TSV; return 0
  fi
  printf "multi\t%s\t%s\t%s\n" "$arm" "$blk" "$m" >> $TSV
  log "$arm b$blk: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}')"
}
summary(){
  python3 -I - $TSV <<'EOF' 2>&1 | tee -a $P
import statistics as st, sys
d = {}; p = {}
for l in open(sys.argv[1]):
    a = l.rstrip("\n").split("\t")
    try:
        x, y = float(a[3]), float(a[4])
    except ValueError:
        continue
    d.setdefault(a[1], []).append(x); p.setdefault(a[1], []).append(y)
b = st.mean(d["base"]) if "base" in d else None
bp = st.mean(p["base"]) if "base" in p else None
for k, x in sorted(d.items(), key=lambda kv: -st.mean(kv[1])):
    rel = f", {100 * (st.mean(x) / b - 1):+.2f} % vs base" if b else ""
    relp = f", {100 * (st.mean(p[k]) / bp - 1):+.2f} %" if bp else ""
    print(f"[hostcc] {k:9s} n={len(x)} decode {st.mean(x):.4f} ({min(x):.4f}-{max(x):.4f}), step {1000 / st.mean(x):.3f} ms{rel}; prefill {st.mean(p[k]):.1f} ({min(p[k]):.1f}-{max(p[k]):.1f}){relp}")
gcc = [v for k in ("base", "gccnoomp") for v in d.get(k, [])]
for k in ("c19", "c24"):
    if k in d and gcc:
        s = "up" if min(d[k]) > max(gcc) else "down" if max(d[k]) < min(gcc) else "level"
        print(f"[hostcc] {k}: {s} against both gcc arms")
EOF
}
for blk in $(seq 1 $N); do
  order=$(python3 -I -c 'import random,sys; a=sys.argv[2:]; random.Random(int(sys.argv[1])).shuffle(a); print(" ".join(a))' "$(( 20261216 + blk ))" $ARMS)
  log "block $blk order: $order"
  for a in $order; do cell $a $blk; done
  log "=== block $blk DONE"
  summary
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore; touch $DONE; log "=== HOSTCC-SCREEN DONE ==="
