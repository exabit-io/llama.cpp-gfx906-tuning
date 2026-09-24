#!/bin/bash
# binrun.sh — bin the Exabit patchsets. For each patchset x: compile x, test x, bin x.
#
# Base: master of exabit-io/mx-llama.cpp = the substrate (mxxm-t fork + llama.cpp v0.5.0) + the patches binned both.
# max-ilp is a BUILD-FLAG patchset: base code, compiled with mixa3607/ML-gfx906's -mllvm -amdgpu-sched-strategy=max-ilp. Each arm = base + one
# patchset, cherry-picked from gfx906-candidates of exabit-io/mx-llama.cpp (our terms with their conflicts already resolved on the substrate). Two patchsets need an
# earlier one to apply, so their arm includes it and they are binned on the INCREMENT over that arm:
#   gdn-producer-fold  needs norm-add-fusion
# The mmvq patchset is ONE unit, 01 15 05 09 (arm name mmvq-batch1-knobs): 01's resolved form uses q8_fast, which
# 05 declares, so 01 cannot compile alone; 15 uses a define only 01 creates and is 01's MUL_MAT_ID correctness fix.
#
# Test (fixed for every arm, nothing else varies): Qwen3.8-27B Q8_0, four dies -sm tensor, -ngl all,
# f16/f16 KV, 125 W/die, RCCL + gated custom AR, llama-batched-bench -ntg 1024.
#   multi-user  axis: 4 x 64K   (4 x 65536)      single-user axis: 1 x 255K (1 x 260864)
# n=5 per arm per axis (n=4's exact-test floor p=0.0286 cannot pass BH q<0.10 over a 24-test family unless
# >=7 tests are real effects; n=5's floor 0.0079 needs 2). Axis-major so multi-user verdicts exist first; inside an axis each block runs
# every arm once in a seeded random order, so drift hits all arms alike. binstats.py bins the result.
set -u
W=/root/night-20260919; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
TL=/root/llama.cpp-benchmarking/tools; CM=$TL/cell-metrics.py; NF=$TL/assert-no-fa-fallback.sh
AB=$TL/assert-build-config.sh; AC=$TL/assert-arms-comparable.sh
WT=/root/wt-bin; BASEREF=exabit-mx/master
P=$W/binrun.progress; TSV=$W/binrun.tsv; DONE=$W/.binrun-done
log(){ echo "$(date -Is) [bin] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE; echo $$ > $W/binrun.pid
ARMS_FILE=$W/binrun-arms.txt
cat > $ARMS_FILE <<'EOF'
base|
norm-add-fusion|1d5918cce 91fdc0277
gdn-producer-fold|1d5918cce 91fdc0277 8a4a9b616 8c99f4af7 c7ee50145 bf65d955c d35a9c793
mmvq-batch1-knobs|4d6246ad7 7af0ac291 df4c199b2 3a49b322e
s1b-repacked-matvec|fd1c2e743 cc46c2333 f9d9662c4
fa-head256-rows|bf4bf4f19 60cad6022
dpp-warp-reductions|49c6ca3b3 c339a4087
max-ilp||-DCMAKE_HIP_FLAGS=-mllvm -amdgpu-sched-strategy=max-ilp
EOF

# ---- phase 1: compile every arm (host only, nothing measuring)
declare -a READY=()
while IFS='|' read -r name commits extra; do
  [ -z "$name" ] && continue
  bd=/root/build-ps-$name
  log "compile $name (${commits:-master head})"
  git -C $WT checkout -qf --detach $BASEREF && git -C $WT clean -qfd
  ok=1
  for c in $commits; do
    if ! git -C $WT cherry-pick --no-commit "$c" >> $W/binrun-build.log 2>&1; then
      log "  $name: cherry-pick $c FAILED onto $BASEREF"; git -C $WT cherry-pick --abort >/dev/null 2>&1; ok=0; break
    fi
  done
  if [ $ok -eq 0 ]; then printf "all\t%s\t-\tAPPLY-FAILED\tAPPLY-FAILED\n" "$name" >> $TSV; continue; fi
  git -C $WT diff --cached > $W/binrun-$name.diff
  rm -rf "$bd" && mkdir -p "$bd"
  {
    cmake -S $WT -B "$bd" -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 \
      -DCMAKE_HIP_ARCHITECTURES=gfx906 -DGGML_HIP_RCCL=ON -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON \
      -DGGML_CUDA_FA_QUANTS=all -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
      ${extra:+"$extra"}
    cmake --build "$bd" -j 24
  } > $W/build-ps-$name.log 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -x "$bd/bin/llama-batched-bench" ]; then
    log "  $name: BUILD FAILED exit=$rc"; grep -m6 -E 'error:' $W/build-ps-$name.log >> $P
    printf "all\t%s\t-\tBUILD-FAILED\tBUILD-FAILED\n" "$name" >> $TSV; continue
  fi
  if ! $AB "$bd" >> $P 2>&1; then log "  $name: violates R3.11 build requirements — not measured"; continue; fi
  if ! strings "$bd"/bin/libggml-hip.so* 2>/dev/null | grep -q GGML_TP_AR_MAX_NE; then
    log "  $name: AR size gate not compiled in — not measured"; continue
  fi
  READY+=("$name"); log "  $name: ready"
done < $ARMS_FILE
git -C $WT checkout -qf --detach $BASEREF && git -C $WT clean -qfd
case " ${READY[*]} " in *" base "*) ;; *) log "FATAL: the base did not build"; touch $DONE; exit 1;; esac
$AC $(for a in "${READY[@]}"; do echo /root/build-ps-$a; done) >> $P 2>&1 \
  || { log "ARMS NOT COMPARABLE — refusing to measure"; touch $DONE; exit 4; }
log "arms comparable: ${READY[*]}"

# ---- phase 2 is gated on the v0.5.0 correctness gate (gate-v050.sh): wait for it by PID, then require a pass
GATE_PID=${1:-}
if [ -n "$GATE_PID" ]; then
  log "waiting for the correctness gate (pid $GATE_PID) before measuring"
  WAIT_MAX=86400 $W/waitproc.sh "$GATE_PID" >> $P 2>&1
fi
GP=$W/gate-build-mx-both.progress
GATE_COMMIT=$(grep -m1 -oE "source commit [0-9a-f]+" $GP 2>/dev/null | awk '{print $3}')
BASE_COMMIT=$(git -C $WT rev-parse $BASEREF)
if [ "$GATE_COMMIT" != "$BASE_COMMIT" ]; then log "GATE IS FOR ${GATE_COMMIT:-nothing}, BASE IS $BASE_COMMIT — not measuring"; touch $DONE; exit 6; fi
tbo=$(grep -c 'test-backend-ops: PASS' $GP 2>/dev/null)
ppl=$(grep -oE 'PPL = [0-9.]+' $GP 2>/dev/null | tail -1 | awk '{print $3}')
fnok=$(grep -cE 'flash-next rc=0, [0-9]{3,} bytes' $GP 2>/dev/null)
pplok=$(awk -v p="${ppl:-0}" 'BEGIN{print (p>=5.55 && p<=5.70)?1:0}')
if [ "${tbo:-0}" -ne 1 ] || [ "$pplok" -ne 1 ] || [ "${fnok:-0}" -ne 1 ]; then
  log "GATE NOT PASSED (test-backend-ops pass=${tbo:-0}, ppl=${ppl:-none}, flash-next ok=${fnok:-0}) — not measuring"
  touch $DONE; exit 6
fi
log "correctness gate passed: test-backend-ops PASS, ppl $ppl, Flash-Next generates"
# ---- phase 2: test
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml GGML_CUDA_ALLREDUCE=nccl
export GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
BPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${BPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/binrun-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/binrun-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/binrun-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/binrun-smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
cell(){ # cell AXIS ARM BLOCK
  local axis=$1; local arm=$2; local blk=$3
  local s=4; local d=65536
  if [ "$axis" = single ]; then s=1; d=260864; fi
  local bd=/root/build-ps-$arm
  local out=$W/br-$axis-$arm-$blk.md
  LD_LIBRARY_PATH=$bd/bin:$bd/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib timeout 7200 \
    $bd/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s \
    > $out 2>${out%.md}.log &
  BPID=$!
  wait $BPID
  BPID=""
  if ! $NF "${out%.md}.log" >/dev/null 2>&1; then
    log "FA FALLBACK in $axis $arm b$blk — discarded"; printf "%s\t%s\t%s\tFALLBACK\tFALLBACK\n" "$axis" "$arm" "$blk" >> $TSV; return 0
  fi
  local m
  if ! m=$(python3 $CM "$out" $d $s 2>>$P); then
    log "CELL FAILED: $axis $arm b$blk"; printf "%s\t%s\t%s\tFAILED\tFAILED\n" "$axis" "$arm" "$blk" >> $TSV; return 0
  fi
  printf "%s\t%s\t%s\t%s\n" "$axis" "$arm" "$blk" "$m" >> $TSV
  log "$axis $arm b$blk: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}')"
}
for axis in multi single; do
  log "=== axis $axis: ${#READY[@]} arms x 5 blocks"
  for blk in 1 2 3 4 5; do
    order=$(python3 -c 'import random,sys; a=sys.argv[2:]; random.Random(int(sys.argv[1])).shuffle(a); print(" ".join(a))' \
      "$(( 20260924 + blk + (${#axis} * 10) ))" "${READY[@]}")
    log "block $blk order: $order"
    for a in $order; do cell $axis $a $blk; done
  done
  log "=== axis $axis DONE"
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $DONE; log "=== BINRUN DONE ==="
