#!/bin/bash
# screen-batch.sh N — screen the next N separable substrate commits as delta-minus units.
#
# 53 of the fork's 138 code commits reverse-apply against the squashed substrate as discrete patches
# (measured 2026-09-20); the other 85 are only separable as 12 feature groups. This screens the
# separable ones: build substrate-MINUS-one-commit, measure against the bundle at the multi-user design
# point, n=2 (the D-4 SCREEN). Screen data never becomes confirmation data.
#
# Every build goes through assert-build-config.sh, so a unit built without RCCL (R3.11) is refused
# rather than measured -- the failure mode that put this whole campaign on the butterfly path.
set -u
W=/root/night-20260919; R=/root/rocm-tests/bench; REPO=/root/exabit-llama.cpp
F=/root/build-faq-allquants   # R3.11: the comparison arm must be FA_QUANTS=all like the minus-builds
M=/root/models/Qwen3.8-27B-Q8_0.gguf
CM=/root/llama.cpp-benchmarking/tools/cell-metrics.py
AB=/root/llama.cpp-benchmarking/tools/assert-build-config.sh
N=${1:-6}
log(){ echo "$(date -Is) [scr] $*" | tee -a $W/screen.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
SEP=/tmp/claude-0/-/56cd63fc-419f-47e8-8261-04392be45e5a/scratchpad/separable.txt
[ -f "$SEP" ] || { echo "FATAL: $SEP missing (the separability measurement)" >&2; exit 1; }
: > $W/screen.tsv 2>/dev/null || true
touch $W/screen.done-units
mapfile -t UNITS < <(cut -f1 "$SEP" | grep -vxFf <(cut -f1 $W/screen.done-units 2>/dev/null || true) | head -$N)
rm -f $W/.screen-done   # clear BEFORE phase 1: a waiter armed during the build phase must not
                        # see the previous run's flag. That returned a waiter instantly on 2026-09-22.
log "=== screening ${#UNITS[@]} separable units"
# ---- phase 1: BUILD (host only, GPU idle)
declare -a READY=()
for sha in "${UNITS[@]}"; do
  log "building minus-$sha"
  if /bin/bash $W/deltaminus.sh commit "$sha" >> $W/screen-build.log 2>&1; then
    bd=/root/build-minus-$(echo $sha | cut -c1-9)
    if $AB "$bd" >/dev/null 2>&1; then READY+=("$sha"); log "  minus-$sha ready"
    else log "  minus-$sha REFUSED: violates R3.11"; fi
  else
    rc=$?
    case $rc in
      4) log "  minus-$sha NOT SEPARABLE or build broke -> bin neutral-required-substrate" ;;
      *) log "  minus-$sha deltaminus exit=$rc" ;;
    esac
    printf '%s\tbuild-failed-rc%s\n' "$sha" "$rc" >> $W/screen.done-units
  fi
done
[ ${#READY[@]} -eq 0 ] && { log "no units built; nothing to measure"; touch $W/.screen-done; exit 0; }
# ---- PRECONDITION: the arms must be comparable BEFORE any GPU time is spent.
# 1.6 GPU hours were thrown away on 2026-09-22 because this check did not exist.
if ! /root/llama.cpp-benchmarking/tools/assert-arms-comparable.sh "$F" \
       $(for sha in "${READY[@]}"; do echo /root/build-minus-$(echo $sha | cut -c1-9); done) \
       >> $W/screen.progress 2>&1; then
  log "ARMS NOT COMPARABLE — refusing to measure. See screen.progress."
  touch $W/.screen-done; exit 4
fi
log "arms verified comparable"
# ---- phase 2: MEASURE (GPU; host now idle)
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml GGML_CUDA_ALLREDUCE=nccl
export GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/screen-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
rm -f $W/.screen-done
QUEUE_PID=$$ DONEFLAG=$W/.screen-done SMCLOG=$W/screen-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/screen-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/screen-smc.log >/dev/null 2>&1 &
sleep 6
cell(){ # cell ARM BUILD REP
  local arm=$1; local bld=$2; local rep=$3; local d=65536; local s=4
  local out=$W/sc-$arm-$rep.md
  LD_LIBRARY_PATH=$bld/bin:$bld/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib timeout 3600 \
    $bld/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s \
    > $out 2>$W/sc-$arm-$rep.log
  local m
  if ! m=$(python3 $CM "$out" "$d" "$s" 2>>$W/screen.progress); then
    log "CELL FAILED: $arm rep$rep"; printf "%s\t%s\tFAILED\tFAILED\n" "$arm" "$rep" >> $W/screen.tsv; return 0
  fi
  printf "%s\t%s\t%s\n" "$arm" "$rep" "$m" >> $W/screen.tsv
  log "$arm rep$rep: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}')"
}
for rep in 1 2; do
  cell bundle $F $rep
  for sha in "${READY[@]}"; do cell "minus-$(echo $sha | cut -c1-9)" "/root/build-minus-$(echo $sha | cut -c1-9)" $rep; done
done
for sha in "${READY[@]}"; do printf '%s\tscreened\n' "$sha" >> $W/screen.done-units; done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.screen-done; log "=== SCREEN BATCH DONE ==="
