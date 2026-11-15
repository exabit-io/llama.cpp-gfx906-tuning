#!/bin/bash
# pkg-swap.sh -- where is the package's 4 x 64K decode deficit against the reference build? rocm-ab2.sh measured the
# installed package at 24.546 tok/s per slot against 24.787 for /root/build-master-v100r (-0.97 %, n = 5 each, every
# package run below every reference run; under the 2 % floor, so the package was accepted). Same source commit, same GPU
# compiler (ROCm 10.0), so the candidates are: the ROCm 10.1 runtime, or the package's build of the host code
# (CMAKE_C_FLAGS/CMAKE_CXX_FLAGS=-march=native on every target; the reference build has it only where GGML_NATIVE puts it;
# stripped; LLAMA_BUILD_TESTS=OFF).
#
# Method: the package's llama-batched-bench with groups of its shared libraries replaced by the reference build's, through
# LD_PRELOAD of all eight libraries by full path (a library that is already loaded satisfies a NEEDED entry by soname, so
# naming all eight leaves nothing to the search path). The ROCm runtime is 10.1 in every swap arm.
#   pkg      the package as installed
#   llama    reference libllama, libllama-common, libllama-batched-bench-impl; package ggml libraries
#   ggml     reference libggml-base, libggml-cpu, libggml; the rest from the package
#   hip      reference libggml-hip; the rest from the package
#   all      all eight from the reference build (= the reference build on the ROCm 10.1 runtime)
#   r100     the reference build on the ROCm 10.0 runtime (rocm-ab2.sh's reference arm)
#   llaman   as llama, but the three libraries from $ALT: the reference configuration built again with only the
#            package's host flags added (EXTRA='-DCMAKE_C_FLAGS=-march=native -DCMAKE_CXX_FLAGS=-march=native' build-rocm.sh)
# Reading: decode per slot at 4 x 64K, the cell of rocm-ab2.sh. An arm that sits with r100 (24.74-24.82 in rocm-ab2) names
# the library group that carries the deficit; pkg's five runs spanned 24.542-24.552. A screen: n = N blocks (default 1),
# arms in seeded random order; no test, the two clusters are 0.24 tok/s apart and 0.01-0.08 wide.
set -u
W=/root/rocm101-migration-20261114; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
TL=/root/llama.cpp-benchmarking/tools; CM=$TL/cell-metrics.py; NF=$TL/assert-no-fa-fallback.sh
N=${N:-1}; ARMS=${ARMS:-"pkg llama ggml hip all r100"}; TAG=${TAG:-pkg-swap}
P=$W/$TAG.progress; TSV=$W/$TAG.tsv; DONE=$W/.$TAG-done; O=$W/$TAG; mkdir -p $O
log(){ echo "$(date -Is) [swap] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE; echo $$ > $W/$TAG.pid
PK=/opt/mx-llama.cpp/lib; RF=/root/build-master-v100r/bin; RT=/opt/rocm/core-10.1; ALT=${ALT:-/root/build-master-v100rn/bin}
LIBS="libggml-base.so.1 libggml.so.1 libggml-cpu.so.1 libggml-hip.so.1 libllama.so.1 libllama-common.so.1 libllama-batched-bench-impl.so"
from_ref(){ # from_ref ARM LIB -> 0 if LIB comes from the reference build in ARM
  case "$1:$2" in
    all:*) return 0;;
    llama:libllama.so.1|llama:libllama-common.so.1|llama:libllama-batched-bench-impl.so) return 0;;
    llaman:libllama.so.1|llaman:libllama-common.so.1|llaman:libllama-batched-bench-impl.so) return 0;;
    ggml:libggml-base.so.1|ggml:libggml-cpu.so.1|ggml:libggml.so.1) return 0;;
    hip:libggml-hip.so.1) return 0;;
  esac
  return 1
}
want(){ # want ARM -> the eight (seven sonames) full paths of the arm, one per line, resolved
  local a=$1 l
  for l in $LIBS; do
    if [ "$a" = llaman ] && from_ref "$a" "$l"; then readlink -f $ALT/$l
    elif [ "$a" = r100 ] || from_ref "$a" "$l"; then readlink -f $RF/$l; else readlink -f $PK/$l; fi
  done
}
preload(){ want "$1" | tr '\n' ':' | sed 's/:$//'; }
arm_env(){ # arm_env ARM -> the environment assignments of the arm, one per line
  case $1 in
    pkg)  echo "LD_LIBRARY_PATH=$PK:$RT/lib";;
    r100) echo "LD_LIBRARY_PATH=$RF:/opt/rocm/core-10.0/lib";;
    *)    echo "LD_LIBRARY_PATH=$RT/lib"; echo "LD_PRELOAD=$(preload "$1")";;
  esac
}
arm_bin(){ if [ "$1" = r100 ]; then echo $RF/llama-batched-bench; else echo /opt/mx-llama.cpp/bin/llama-batched-bench; fi; }
# static check of every arm: the loader must map exactly the arm's libraries, and one ROCm tree
[ "$(git -C /root/wt-master-v100 rev-parse HEAD)" = "$(cat /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT)" ] \
  || { log "FATAL: the package and the reference build are not the same commit"; touch $DONE; exit 1; }
for a in $ARMS; do
  case $a in pkg|llama|llaman|ggml|hip|all|r100) ;; *) log "FATAL: unknown arm $a"; touch $DONE; exit 1;; esac
  mapfile -t E < <(arm_env $a)
  got=$(env "${E[@]}" ldd "$(arm_bin $a)" 2>/dev/null | awk '{ if ($2 == "=>") print $3; else print $1 }' | grep -E '/lib(ggml|llama)' | xargs -r -n1 readlink -f | sort -u)
  exp=$(want $a | sort -u)
  [ "$got" = "$exp" ] || { log "FATAL: arm $a maps other libraries than intended: $(echo "$got" | tr '\n' ' ')"; touch $DONE; exit 1; }
  rtw=/opt/rocm/core-10.1; [ "$a" = r100 ] && rtw=/opt/rocm/core-10.0
  nro=$(env "${E[@]}" ldd "$(arm_bin $a)" | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | wc -l)
  bad=$(env "${E[@]}" ldd "$(arm_bin $a)" | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | grep -vc "^$rtw/")
  [ "$bad" = 0 ] && [ "$nro" -gt 5 ] || { log "FATAL: arm $a resolves $bad of $nro ROCm libraries outside $rtw"; touch $DONE; exit 1; }
  log "arm $a: $(echo "$exp" | sed -e "s|$RF/|ref:|" -e "s|$PK/|pkg:|" -e "s|$ALT/|alt:|" -e 's/\.so[.0-9]*$//' | tr '\n' ' '); $nro ROCm libraries under $rtw"
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
  local out=$O/multi-$arm-$blk.md E child got exp
  mapfile -t E < <(arm_env $arm)
  env "${E[@]}" AMD_COMGR_CACHE_DIR=/root/.cache/comgr-swap-$arm timeout 7200 \
    "$(arm_bin $arm)" -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
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
for blk in $(seq 1 $N); do
  order=$(python3 -I -c 'import random,sys; a=sys.argv[2:]; random.Random(int(sys.argv[1])).shuffle(a); print(" ".join(a))' "$(( 20261115 + blk ))" $ARMS)
  log "block $blk order: $order"
  for a in $order; do cell $a $blk; done
done
log "decode per slot by arm: $(for a in $ARMS; do printf '%s [%s] ' "$a" "$(awk -F'\t' -v a="$a" '$2==a && $4+0>0 {printf "%s%.4f", (n++ ? " " : ""), $4}' $TSV)"; done)"
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore; touch $DONE; log "=== PKG-SWAP DONE ==="
