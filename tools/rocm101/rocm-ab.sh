#!/bin/bash
# rocm-ab.sh -- ROCm 10.0 against ROCm 10.1 at the two design points, one source commit (mx-llama.cpp master df2c6e5c5),
# the survey harness's cell (night-20260919/binrun.sh phase 2) with two arms:
#   r100  /root/build-master-v100r  built by build-rocm.sh against /opt/rocm/core-10.0, run on core-10.0's libraries
#   r101  /root/build-master-v101   built by build-rocm.sh against /opt/rocm/core-10.1, run on core-10.1's libraries
# Cell (fixed): Qwen3.8-27B Q8_0, four dies -sm tensor, -ngl all, f16/f16 KV, 125 W per die (macpro-fand pin),
# RCCL + gated custom AR, llama-batched-bench -ntg 1024.   multi: 4 x 64K (4 x 65536)   single: 1 x 255K (1 x 260864)
#
# RULE, WRITTEN BEFORE ANY CELL RAN (the survey's rule, binstats.py, with a family of this run's four tests):
#   n = 5 runs per arm per axis; inside a block both arms run once in a seeded random order
#   statistic: multi-user MEAN, single-user MEDIAN of the n runs; effect = r101 / r100 - 1
#   test: two-sided exact permutation, 5 vs 5 (252 arrangements, floor 0.0079)
#   family: 2 axes x 2 metrics (decode per slot, prefill) = 4 tests, Benjamini-Hochberg q < 0.10
#   an effect counts only if q < 0.10 AND |effect| >= 2 %
#   ROCm 10.1 is ACCEPTED for the packages if no metric on either axis regresses under that rule.
# Waits for the host to be idle of builds: start it only when nothing else is compiling or measuring.
set -u
W=/root/rocm101-migration-20261114; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
TL=/root/llama.cpp-benchmarking/tools; CM=$TL/cell-metrics.py; NF=$TL/assert-no-fa-fallback.sh
N=${N:-5}; AXES=${AXES:-"multi single"}
P=$W/rocm-ab.progress; TSV=$W/rocm-ab.tsv; DONE=$W/.rocm-ab-done; O=$W/rocm-ab; mkdir -p $O
log(){ echo "$(date -Is) [ab] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE; echo $$ > $W/rocm-ab.pid
declare -A BD=([r100]=/root/build-master-v100r [r101]=/root/build-master-v101)
declare -A RT=([r100]=/opt/rocm/core-10.0 [r101]=/opt/rocm/core-10.1)
ARMS="r100 r101"
# every ROCm library of an arm must come from that arm's tree, and both arms must be the same source commit
for a in $ARMS; do
  bd=${BD[$a]}; rt=${RT[$a]}
  [ -x $bd/bin/llama-batched-bench ] || { log "FATAL: $a has no llama-batched-bench"; touch $DONE; exit 1; }
  wrong=$(LD_LIBRARY_PATH=$bd/bin:$rt/lib ldd $bd/bin/libggml-hip.so $bd/bin/llama-batched-bench | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | grep -vc "^$rt/")
  n=$(LD_LIBRARY_PATH=$bd/bin:$rt/lib ldd $bd/bin/libggml-hip.so | awk '$3 ~ /^\/opt\/rocm/' | wc -l)
  [ "$wrong" = 0 ] && [ "$n" -gt 5 ] || { log "FATAL: $a resolves $wrong of its ROCm libraries outside $rt ($n found)"; touch $DONE; exit 1; }
  log "$a: $bd, $n ROCm libraries, all under $rt; hip $(LD_LIBRARY_PATH=$bd/bin:$rt/lib ldd $bd/bin/libggml-hip.so | awk '/libamdhip64/ {print $3}' | xargs readlink -f | sed 's/.*libamdhip64.so.//'); source $(grep -m1 -oE 'CMAKE_HOME_DIRECTORY:INTERNAL=.*' $bd/CMakeCache.txt | cut -d= -f2)"
  strings $bd/bin/libggml-hip.so | grep -q GGML_TP_AR_MAX_NE || { log "FATAL: $a has no AR size gate"; touch $DONE; exit 1; }
done
[ "$(git -C /root/wt-master-v100 rev-parse --short HEAD)" = df2c6e5c5 ] && [ -z "$(git -C /root/wt-master-v100 status --porcelain --untracked-files=no)" ] \
  || { log "FATAL: the source worktree is not a clean df2c6e5c5"; touch $DONE; exit 1; }
for a in $ARMS; do grep -E '^(GGML_HIP_RCCL|GGML_CUDA_FA_QUANTS|GGML_HIP_GRAPHS|GGML_NATIVE|GGML_HIP_GFX906_MAX_ILP|CMAKE_BUILD_TYPE)[:=]' ${BD[$a]}/CMakeCache.txt | sort > $O/cache-$a.txt; done
cmp -s $O/cache-r100.txt $O/cache-r101.txt || { log "FATAL: the arms' build options differ: $(diff $O/cache-r100.txt $O/cache-r101.txt | tr '\n' ' ')"; touch $DONE; exit 1; }
log "arms comparable: $(tr '\n' ' ' < $O/cache-r101.txt | cut -c1-300); n=$N, axes: $AXES; load $(cut -d' ' -f1-3 /proc/loadavg)"

. $R/gpu-test-env.sh
export GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
BPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${BPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/rocm-ab-clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/rocm-ab-smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $W/rocm-ab-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/rocm-ab-smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
log "$(state_line)"
cell(){ # cell AXIS ARM BLOCK
  local axis=$1 arm=$2 blk=$3 s=4 d=65536
  if [ "$axis" = single ]; then s=1; d=260864; fi
  local bd=${BD[$arm]} rt=${RT[$arm]} out=$O/$axis-$arm-$blk.md
  env LD_LIBRARY_PATH=$bd/bin:$rt/lib AMD_COMGR_CACHE_DIR=/root/.cache/comgr-ab-$arm timeout 7200 \
    $bd/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s \
    > $out 2>${out%.md}.log &
  BPID=$!
  wait $BPID
  local rc=$?
  BPID=""
  if ! $NF "${out%.md}.log" >/dev/null 2>&1; then
    log "FA FALLBACK in $axis $arm b$blk -- discarded"; printf "%s\t%s\t%s\tFALLBACK\tFALLBACK\n" "$axis" "$arm" "$blk" >> $TSV; return 0
  fi
  if grep -qiE 'NCCL init failed|probe AllReduce failed' "${out%.md}.log"; then
    log "RCCL NOT UP in $axis $arm b$blk -- discarded"; printf "%s\t%s\t%s\tNORCCL\tNORCCL\n" "$axis" "$arm" "$blk" >> $TSV; return 0
  fi
  local m
  if ! m=$(python3 -I $CM "$out" $d $s 2>>$P); then
    log "CELL FAILED (rc=$rc): $axis $arm b$blk"; printf "%s\t%s\t%s\tFAILED\tFAILED\n" "$axis" "$arm" "$blk" >> $TSV; return 0
  fi
  printf "%s\t%s\t%s\t%s\n" "$axis" "$arm" "$blk" "$m" >> $TSV
  log "$axis $arm b$blk: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}')"
}
for axis in $AXES; do
  log "=== axis $axis: 2 arms x $N blocks"
  for blk in $(seq 1 $N); do
    order=$(python3 -I -c 'import random,sys; a=sys.argv[2:]; random.Random(int(sys.argv[1])).shuffle(a); print(" ".join(a))' "$(( 20261114 + blk + (${#axis} * 10) ))" $ARMS)
    log "block $blk order: $order"
    for a in $order; do cell $axis $a $blk; done
  done
  log "=== axis $axis DONE"
  python3 -I $W/rocm-ab-stats.py $TSV 2>&1 | tee -a $P
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore; touch $DONE; log "=== ROCM-AB DONE ==="
