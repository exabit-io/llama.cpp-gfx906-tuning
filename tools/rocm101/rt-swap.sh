#!/bin/bash
# rt-swap.sh -- which library of the ROCm 10.1 runtime carries its 4 x 64K decode cost? pkg-swap.sh measured the same
# binaries (the reference build /root/build-master-v100r, compiled with the ROCm 10.0 toolchain) at 24.805 tok/s per slot
# on the ROCm 10.0 runtime and 24.659 on the 10.1 runtime (-0.6 %, n = 4 each, ranges disjoint). The lead asked for the
# cause (2026-11-15). The machine has changed since (other cards in slots 5-7, pci=pcie_bus_perf, a fresh boot: the same
# controls now read about 0.8 % higher), so both controls are in every block and nothing is compared across sessions.
#
# Method: the reference build's llama-batched-bench and its own libraries in every arm; only the ROCm libraries differ.
# The ROCm libraries carry DT_RPATH=$ORIGIN, so one of them preloaded alone drags its whole tree in after it. Every arm
# but plain101 therefore preloads ALL ROCm libraries of the process by full path (the list ldd gives for the 10.1
# runtime), each taken from the tree the arm names; a library that is already loaded satisfies a NEEDED entry or a dlopen
# by soname, so nothing is left to RPATH or the search path.
#   rt101     every ROCm library from the 10.1 tree                   rt100   every one from the 10.0 tree
#   plain101  the 10.1 runtime through LD_LIBRARY_PATH alone, no preload (shows whether the preload itself costs anything)
#   rccl100   10.1, but librccl.so.1 from 10.0 (RCCL 2.30.4 instead of 2.30.7)
#   hip100    10.1, but libamdhip64.so.7 from 10.0 (HIP 7.15 instead of 7.16)
#   rocr100   10.1, but libhsa-runtime64.so.1 from 10.0
#   blas100   10.1, but the BLAS stack from 10.0 (rocblas, hipblas, hipblaslt, rocsolver, origami, rocroller)
#   comgr100  10.1, but libamd_comgr.so.3 from 10.0 with its libLLVM and libclang-cpp 23
#   misc100   10.1, but everything not named above from 10.0 (rocprofiler-register, roctx, amd_smi, rocm-core, kpack, sysdeps)
# Each cell compares the ROCm files in /proc/<pid>/maps with the arm's list, file by file.
# Cell: rocm-ab2.sh's multi-user cell (Qwen3.8-27B Q8_0, 4 x 64K, -sm tensor, f16 KV, 125 W, RCCL + gated custom AR,
# -ntg 1024). DEPTH/NTG/MAPWAIT shrink it for a start-up smoke run of the mixed arms (never a result: below the 64K floor).
#
# READING, WRITTEN BEFORE THE FIRST CELL: a screen, n = N blocks (default 3), every arm once per block in a seeded random
# order, statistic the mean decode per slot. (1) If the ranges of rt100 and rt101 overlap, the runtime difference is not
# reproduced in this machine state and no arm is read. (2) Otherwise an arm "carries the cost" when its range overlaps
# rt100's and not rt101's (taking that one library from 10.0 restores the speed); it is "not it" when its range overlaps
# rt101's and not rt100's; anything else is unresolved. (3) If no single group carries it, the cost is in an
# interaction or is spread, and pairs are the next screen. (4) plain101 is expected to match rt101; if it does not, the
# preload mechanism itself moves decode and the arms are read against rt101 only. Prefill is reported, not tested.
set -u
W=/root/rocm101-migration-20261114; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
TL=/root/llama.cpp-benchmarking/tools; CM=$TL/cell-metrics.py; NF=$TL/assert-no-fa-fallback.sh
N=${N:-3}; ARMS=${ARMS:-"rt101 rt100 plain101 rccl100 hip100 rocr100 blas100 comgr100 misc100"}; TAG=${TAG:-rt-swap}
DEPTH=${DEPTH:-65536}; NTG=${NTG:-1024}; MAPWAIT=${MAPWAIT:-45}
P=$W/$TAG.progress; TSV=$W/$TAG.tsv; DONE=$W/.$TAG-done; O=$W/$TAG; mkdir -p $O
log(){ echo "$(date -Is) [rt] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE; echo $$ > $W/$TAG.pid
RF=/root/build-master-v100r/bin; A=/opt/rocm/core-10.0/lib; B=/opt/rocm/core-10.1/lib; EXE=$RF/llama-batched-bench
SONAMES=$(LD_LIBRARY_PATH=$RF:$B ldd $EXE | awk '$2 == "=>" && $3 ~ /^\/opt\/rocm\// {print $1}' | sort -u)
G_RCCL="librccl.so.1"; G_HIP="libamdhip64.so.7"; G_ROCR="libhsa-runtime64.so.1"; G_COMGR="libamd_comgr.so.3"
G_BLAS="librocblas.so.5 libhipblas.so.3 libhipblaslt.so.1 librocsolver.so.0 liborigami.so.1 librocroller.so.1"
has(){ [[ " $1 " == *" $2 "* ]]; }
in100(){ # in100 ARM SONAME -> 0 when the arm takes SONAME from the 10.0 tree
  case $1 in
    rt100) return 0;; rt101|plain101) return 1;;
    rccl100) has "$G_RCCL" $2;; hip100) has "$G_HIP" $2;; rocr100) has "$G_ROCR" $2;; blas100) has "$G_BLAS" $2;;
    comgr100) has "$G_COMGR" $2;;
    misc100) ! has "$G_RCCL $G_HIP $G_ROCR $G_BLAS $G_COMGR" $2;;
    *) return 1;;
  esac
}
paths(){ # paths ARM -> the full resolved path of every ROCm library of the arm, one per line
  local arm=$1 s
  for s in $SONAMES; do
    case $s in
      libLLVM.so.*|libclang-cpp.so.*)    # comgr's own LLVM: the version of the tree comgr comes from
        if in100 $arm libamd_comgr.so.3; then readlink -f $A/${s%%.so.*}.so.23.0git; else readlink -f $B/$s; fi;;
      *) if in100 $arm $s; then readlink -f $A/$s; else readlink -f $B/$s; fi;;
    esac
  done
}
arm_env(){ # arm_env ARM -> environment assignments, one per line
  echo "LD_LIBRARY_PATH=$RF:$B"
  [ "$1" = plain101 ] || echo "LD_PRELOAD=$(paths $1 | paste -sd:)"
}
# check_maps ARM FILE -> 0 when the ROCm files mapped (FILE, one resolved path per line) are exactly the arm's
check_maps(){
  if [ "$1" = plain101 ]; then [ "$(grep -c '^/opt/rocm/core-10.0/' $2)" = 0 ] && [ "$(grep -c '^/opt/rocm/core-10.1/' $2)" -gt 5 ]
  else [ "$(sort -u $2)" = "$(paths $1 | sort -u)" ]; fi
}
[ "$(echo "$SONAMES" | wc -l)" -gt 20 ] || { log "FATAL: ldd names only $(echo "$SONAMES" | wc -l) ROCm libraries"; touch $DONE; exit 1; }
for a in $ARMS; do
  for f in $(paths $a); do [ -f "$f" ] || { log "FATAL: arm $a: no $f"; touch $DONE; exit 1; }; done
  log "arm $a: from 10.0: $(paths $a | grep -c "^${A%/lib}/") of $(paths $a | wc -l) ROCm libraries: $(paths $a | grep "^${A%/lib}/" | xargs -r -n1 basename | tr '\n' ' ' | cut -c1-160)"
done
[ -x $EXE ] || { log "FATAL: no $EXE"; touch $DONE; exit 1; }
log "n=$N, arms: $ARMS; depth $DEPTH, ntg $NTG; boot $(uptime -s); cmdline $(grep -oE 'pci=[^ ]+' /proc/cmdline | tr '\n' ' '); load $(cut -d' ' -f1-3 /proc/loadavg)"

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
  local arm=$1 blk=$2 s=4 d=$DEPTH
  local out=$O/multi-$arm-$blk.md child m rc
  # shellcheck disable=SC2046
  timeout 7200 env $(arm_env $arm) AMD_COMGR_CACHE_DIR=/root/.cache/comgr-rt-$arm \
    $EXE -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg $NTG -npl $s \
    > $out 2>${out%.md}.log &
  BPID=$!
  sleep $MAPWAIT      # the libraries are mapped long before; the model is still loading
  child=$(pgrep -P $BPID | head -1)
  awk '$6 ~ /^\/opt\/rocm\// {print $6}' /proc/${child:-0}/maps 2>/dev/null | xargs -r -n1 readlink -f | sort -u > ${out%.md}.maps
  wait $BPID
  rc=$?
  BPID=""
  if ! check_maps $arm ${out%.md}.maps; then
    log "WRONG LIBRARIES in $arm b$blk -- discarded: mapped but not listed: $(comm -23 <(sort -u ${out%.md}.maps) <(paths $arm | sort -u) | tr '\n' ' ' | cut -c1-300); listed but not mapped: $(comm -13 <(sort -u ${out%.md}.maps) <(paths $arm | sort -u) | xargs -r -n1 basename | tr '\n' ' ' | cut -c1-200)"
    printf "multi\t%s\t%s\tWRONGLIBS\tWRONGLIBS\n" "$arm" "$blk" >> $TSV; return 0
  fi
  if ! $NF "${out%.md}.log" >/dev/null 2>&1; then
    log "FA FALLBACK in $arm b$blk -- discarded"; printf "multi\t%s\t%s\tFALLBACK\tFALLBACK\n" "$arm" "$blk" >> $TSV; return 0
  fi
  if grep -qiE 'NCCL init failed|probe AllReduce failed' "${out%.md}.log"; then
    log "RCCL NOT UP in $arm b$blk -- discarded"; printf "multi\t%s\t%s\tNORCCL\tNORCCL\n" "$arm" "$blk" >> $TSV; return 0
  fi
  if ! m=$(python3 -I $CM "$out" $d $s 2>>$P); then
    log "CELL FAILED (rc=$rc): $arm b$blk: $(tail -2 ${out%.md}.log | tr '\n' ' ' | cut -c1-200)"; printf "multi\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$blk" >> $TSV; return 0
  fi
  printf "multi\t%s\t%s\t%s\n" "$arm" "$blk" "$m" >> $TSV
  log "$arm b$blk: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}')"
}
summary(){
  python3 -I - $TSV <<'EOF' 2>&1 | tee -a $P
import statistics as st, sys
d = {}
for l in open(sys.argv[1]):
    a = l.rstrip("\n").split("\t")
    try: d.setdefault(a[1], []).append((float(a[3]), float(a[4])))
    except ValueError: pass
b = st.mean(x[0] for x in d["rt101"]) if "rt101" in d else None
for k, x in sorted(d.items(), key=lambda kv: -st.mean(v[0] for v in kv[1])):
    dec = [v[0] for v in x]; pre = [v[1] for v in x]
    rel = f", {100 * (st.mean(dec) / b - 1):+.2f} % vs rt101" if b else ""
    print(f"[rt] {k:9s} n={len(x)} decode {st.mean(dec):.4f} ({min(dec):.4f}-{max(dec):.4f}), step {1000 / st.mean(dec):.3f} ms{rel}; prefill {st.mean(pre):.1f}")
EOF
}
for blk in $(seq 1 $N); do
  order=$(python3 -I -c 'import random,sys; a=sys.argv[2:]; random.Random(int(sys.argv[1])).shuffle(a); print(" ".join(a))' "$(( 20261150 + blk ))" $ARMS)
  log "block $blk order: $order"
  for a in $order; do cell $a $blk; done
  log "=== block $blk DONE"
  summary
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore; touch $DONE; log "=== RT-SWAP DONE ==="
