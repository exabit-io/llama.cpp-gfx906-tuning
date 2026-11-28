#!/bin/bash
# screen.sh TAG D S NTG N ARM=BUILD:ROCMTREE[:PRELOAD] ... -- the survey cell at a chosen depth (D tokens per slot, S slots, NTG decode
# tokens), N runs per arm, arms run in the given order within each block. For screening candidate builds before the
# full design-point A/B: same model, flags and environment as rocm-ab.sh. Appends to screen-TAG.{progress,tsv}.
# PRELOAD (optional): a libggml-hip.so to load instead of the build's own (object-swap experiments, swaplink.py).
# SHUFFLE=1 in the environment: a fresh random arm order in every block (logged), for confirmation passes.
# KVK / KVV in the environment: the KV cache types (-ctk / -ctv), default f16 f16. ARMKV="arm=K/V ..." gives single arms
# their own types (an f16 arm beside FP8 arms).
# ARMENV="arm:VAR=VALUE ..." in the environment: VAR=VALUE is set for that arm's runs only, after the common exports
# (arms that share one build and differ in a runtime switch). Logged at the start.
set -u
W=/root/rebase-v110; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
TL=/root/llama.cpp-benchmarking/tools; CM=$TL/cell-metrics.py; NF=$TL/assert-no-fa-fallback.sh
TAG=$1; D=$2; S=$3; NTG=$4; N=$5; shift 5
KVK=${KVK:-f16}; KVV=${KVV:-f16}
P=$W/screen-$TAG.progress; TSV=$W/screen-$TAG.tsv; DONE=$W/.screen-$TAG-done; O=$W/screen-$TAG; mkdir -p $O
log(){ echo "$(date -Is) [screen] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE
declare -A BD=() RT=() PL=(); ARMS=""
for spec in "$@"; do
  a=${spec%%=*}; rest=${spec#*=}; bd=${rest%%:*}; rest=${rest#*:}; rt=${rest%%:*}; pl=""; [ "$rest" != "$rt" ] && pl=${rest#*:}
  [ -z "$pl" ] || [ -f "$pl" ] || { log "FATAL: $a preload $pl missing"; touch $DONE; exit 1; }
  wrong=$(LD_LIBRARY_PATH=$bd/bin:$rt/lib ldd $bd/bin/libggml-hip.so $bd/bin/llama-batched-bench | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | grep -vc "^$rt/")
  n=$(LD_LIBRARY_PATH=$bd/bin:$rt/lib ldd $bd/bin/libggml-hip.so | awk '$3 ~ /^\/opt\/rocm/' | wc -l)
  [ "$wrong" = 0 ] && [ "$n" -gt 5 ] || { log "FATAL: $a resolves $wrong ROCm libraries outside $rt ($n found)"; touch $DONE; exit 1; }
  BD[$a]=$bd; RT[$a]=$rt; PL[$a]=$pl; ARMS="$ARMS $a"
  log "$a: $bd on $rt${pl:+ with $pl preloaded}"
done
[ "$KVK/$KVV" = f16/f16 ] && [ -z "${ARMKV:-}" ] || log "KV cache types: $KVK / $KVV${ARMKV:+; per arm: $ARMKV}"
[ -z "${ARMENV:-}" ] || log "per-arm environment: $ARMENV"
. $R/gpu-test-env.sh
export GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
BPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${BPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/screen-$TAG-clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/screen-$TAG-smc.log SCLK_GUARD=${SCLK_GUARD:-1} nohup $R/clamp-watchdog-v2.sh >> $W/screen-$TAG-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/screen-$TAG-smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
for blk in $(seq 1 $N); do
  order=$ARMS
  if [ "${SHUFFLE:-0}" = 1 ]; then order=$(echo $ARMS | tr ' ' '\n' | shuf | tr '\n' ' '); log "block $blk order: $order"; fi
  for arm in $order; do
    bd=${BD[$arm]}; rt=${RT[$arm]}; pl=${PL[$arm]}; out=$O/$arm-$blk.md
    kk=$KVK; kv=$KVV; for x in ${ARMKV:-}; do [ "${x%%=*}" = "$arm" ] && { kk=${x#*=}; kv=${kk#*/}; kk=${kk%%/*}; }; done
    ae=""; for x in ${ARMENV:-}; do [ "${x%%:*}" = "$arm" ] && ae="$ae ${x#*:}"; done
    env $ae ${pl:+LD_PRELOAD=$pl} LD_LIBRARY_PATH=$bd/bin:$rt/lib AMD_COMGR_CACHE_DIR=/root/.cache/comgr-screen-$arm timeout 7200 \
      $bd/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
      -fa on -ctk $kk -ctv $kv -b 2048 -ub 2048 -c $(( S*(D+1280) )) -npp $D -ntg $NTG -npl $S \
      > $out 2>${out%.md}.log &
    BPID=$!
    wait $BPID
    BPID=""
    $NF "${out%.md}.log" >/dev/null 2>&1 || { log "FA FALLBACK in $arm b$blk"; continue; }
    grep -qiE 'NCCL init failed|probe AllReduce failed' "${out%.md}.log" && { log "RCCL NOT UP in $arm b$blk"; continue; }
    if ! m=$(python3 -I $CM "$out" $D $S 2>>$P); then log "CELL FAILED: $arm b$blk"; continue; fi
    printf "%s\t%s\t%s\t%s\n" "$S x $D" "$arm" "$blk" "$m" >> $TSV
    log "$S x $D $arm b$blk: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}')"
  done
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore; touch $DONE; log "=== SCREEN $TAG DONE ==="
