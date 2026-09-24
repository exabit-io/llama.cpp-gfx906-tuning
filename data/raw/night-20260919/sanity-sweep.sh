#!/bin/bash
# sanity-sweep.sh — regression sanity for the installed master build (lead, 2026-09-24).
#
# Three builds, same flags: exabit-io/mx-llama.cpp master (a23e12438, installed at /opt/llama.cpp-mx), the pure substrate
# merge-v0.5.0 (528384980, mxxm-t's fork + v0.5.0, without our commits) and stock llama.cpp v0.5.0 (7fe450e); every
# /root/models/*Q8*.gguf (lead: Q8 only), at 4x64K and 1x255K, f16/f16 KV, -sm tensor, the usual switches.
# n=1 per build per cell: a sanity sweep (does every model run, and is master anywhere clearly slower than stock), not a
# bin. Each model's three builds run back to back, order rotated per model so drift does not favour one build.
set -u
W=/root/night-20260919; R=/root/rocm-tests/bench; TL=/root/llama.cpp-benchmarking/tools
CM=$TL/cell-metrics.py; NF=$TL/assert-no-fa-fallback.sh; AB=$TL/assert-build-config.sh; AC=$TL/assert-arms-comparable.sh
P=$W/sweep.progress; TSV=$W/sweep.tsv; DONE=$W/.sweep-done; O=$W/sweep
MX=/opt/llama.cpp-mx; MXB=/root/build-mx-both; SU=/root/build-mx-sync; ST=/root/build-stock-v050-rccl
log(){ echo "$(date -Is) [sweep] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE; mkdir -p $O; echo $$ > $W/sweep.pid
PRED=${1:-}
if [ -n "$PRED" ]; then
  log "waiting for the stock v0.5.0 build (pid $PRED)"
  WAIT_MAX=7200 $W/waitproc.sh "$PRED" >> $P 2>&1
fi
if [ ! -x $ST/bin/llama-batched-bench ]; then log "FATAL: the stock build did not produce llama-batched-bench"; touch $DONE; exit 1; fi
if ! $AB $MXB >> $P 2>&1; then log "FATAL: master build violates R3.11"; touch $DONE; exit 1; fi
if ! $AB $ST >> $P 2>&1; then log "FATAL: stock build violates R3.11"; touch $DONE; exit 1; fi
if ! $AB $SU >> $P 2>&1; then log "FATAL: substrate build violates R3.11"; touch $DONE; exit 1; fi
if [ "$(git -C /root/wt-mx-sync rev-parse --short=9 HEAD)" != 528384980 ]; then log "FATAL: substrate build is not 528384980"; touch $DONE; exit 1; fi
if ! $AC $MXB $SU $ST >> $P 2>&1; then log "FATAL: builds differ in a code-affecting option"; touch $DONE; exit 4; fi
if [ "$(readlink -f $MX)" != /opt/llama.cpp-mx-a23e12438 ]; then log "FATAL: /opt/llama.cpp-mx is not the checked install"; touch $DONE; exit 1; fi
log "builds checked: master $(readlink -f $MX) (from $MXB), substrate $SU (528384980), stock $ST (7fe450e) — RCCL + FA_QUANTS=all in all three"
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml GGML_CUDA_ALLREDUCE=nccl
export GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
BPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${BPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $O/clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$O/smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $O/watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $O/smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
# SKIP_MODELS="m1 m2": resume after a stop — those models are already in sweep.tsv, keep it and skip them (rotation index
# still advances, so every model keeps the build order it would have had in one uninterrupted run).
[ -n "${SKIP_MODELS:-}" ] || : > $TSV
cell(){ # cell BUILD MODEL SLOTS DEPTH
  local b=$1; local m=$2; local s=$3; local d=$4
  local bin=$MX/bin; local lib=$MX/lib
  if [ "$b" = stock ]; then bin=$ST/bin; lib=$ST/bin:$ST/lib; fi
  if [ "$b" = substrate ]; then bin=$SU/bin; lib=$SU/bin:$SU/lib; fi
  local mn; mn=$(basename "$m" .gguf)
  local cellname="${s}x$((d/1024))K"
  local out=$O/$mn-$cellname-$b.md
  LD_LIBRARY_PATH=$lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib timeout 7200 \
    $bin/llama-batched-bench -m "$m" --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s \
    > "$out" 2> "${out%.md}.log" < /dev/null &
  BPID=$!
  wait $BPID
  local rc=$?
  BPID=""
  local st=ok; local m1="-"; local m2="-"
  if [ $rc -ne 0 ]; then
    st="FAILED(rc=$rc)"
  elif ! $NF "${out%.md}.log" >/dev/null 2>&1; then
    st="FA-FALLBACK"
  else
    local met
    if met=$(python3 $CM "$out" $d $s 2>>$P); then
      m1=$(echo "$met" | cut -f1); m2=$(echo "$met" | cut -f2)
    else
      st="NO-RESULT"
    fi
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$mn" "$cellname" "$b" "$st" "$m1" "$m2" >> $TSV
  log "$mn $cellname $b: $st decode=$m1 prefill=$m2"
}
MODELS="Qwen3.8-27B-Q8_0 Qwen3.5-0.8B-Q8_0 Qwen3.5-2B-Q8_0 Qwen2.5-1.5B-Instruct-Q8_0 Qwen3.5-9B-Q8_0 Qwen3.5-35B-A3B-Q8_0"
n_found=$(ls /root/models/*Q8*.gguf | wc -l)
log "=== sweeping $(echo $MODELS | wc -w) models (of $n_found /root/models/*Q8*.gguf) x 2 cells x 3 builds"
i=0
for mn in $MODELS; do
  m=/root/models/$mn.gguf
  case " ${SKIP_MODELS:-} " in *" $mn "*) log "$mn done in an earlier run — skipped"; i=$((i+1)); continue;; esac
  if [ ! -f "$m" ]; then log "missing $m — skipped"; continue; fi
  case $((i % 3)) in
    0) order="stock substrate master";;
    1) order="substrate master stock";;
    2) order="master stock substrate";;
  esac
  for cfg in "4 65536" "1 260864"; do
    set -- $cfg
    for b in $order; do cell $b "$m" $1 $2; done
  done
  i=$((i+1))
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $DONE; log "=== SWEEP DONE ==="
