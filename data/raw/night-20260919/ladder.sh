#!/bin/bash
# Capacity ladder — REQUIREMENTS R3.1 / R3.2 / §5.2.
# Where does >= 12 tok/s per request hold across slots x depth, q8_0 KV, on the
# ROCm 10.0 multi-user build?  llama-batched-bench, one D-token prompt per sequence.
#
# Methodology: /root/rocm-tests/bench/service-A.sh section A1, with three changes:
#   -ngl all     explicit.  Build 11067 defaults -ngl to "auto", which at these depths
#                can silently leave layers on the host and report a decode number that
#                looks like a result instead of an OOM (trap T5b).
#   libs pinned  LD_LIBRARY_PATH=$P/lib:$RT  (trap T1 — no RUNPATH on any binary here).
#   VRAM peak    sampled per cell, for R3.2's 31 GiB/die budget.
#
# Env in:  CELLS (slots:depth list, q8_0 KV assumed), GPU_CAP (W/die), TAG.
# Nothing in this file is edited while it runs (trap: bash reads scripts incrementally).
set -u
TAG=${TAG:-ladder-q8kv-200w}
GPU_CAP=${GPU_CAP:-200}
CELLS=${CELLS:-"4:65536 8:196608 4:262144 8:65536 4:131072 8:98304 8:131072 4:196608"}
W=/root/night-20260919; B=/root/rocm-tests/bench; Q=$B/bench-queue.progress
P=/opt/llama.cpp-gfx906-rocm10; RT=/opt/rocm/core-10.0/lib
M=/root/models/Qwen3.8-27B-Q8_0.gguf
RAPLD=/sys/class/powercap/intel-rapl:0
OUT=$W/$TAG.md; LOGD=$W/$TAG-logs; DONEFLAG=$W/.$TAG-done
mkdir -p $LOGD; rm -f $DONEFLAG
log() { echo "$(date -Is) [$TAG] $*" | tee -a $W/$TAG.progress >> $Q; }
kpid() { [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100

. $B/gpu-test-env.sh
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
export LD_LIBRARY_PATH=$P/lib:$RT
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all"
DIES="0b 0e 1b 1e"

cleanup() {   # trap MUST end in exit (trap T3) or bash clears traps and resumes the script
  trap - INT TERM EXIT
  pkill -x llama-batched-bench 2>/dev/null
  kpid "${SAMP:-}"; kpid "${VS:-}"
  pkill -f "clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "smc-log[.]sh $W" 2>/dev/null
  for d in $DIES; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1
}
trap cleanup INT TERM EXIT

setmax                                   # fans max, perf high, host RAPL 150 W (the 1228 W envelope)
start_sampler $W/$TAG-clocks.txt
for d in $DIES; do echo $((GPU_CAP*1000000)) > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE="^/bin/bash $W/ladder[.]sh" DONEFLAG=$DONEFLAG SMCLOG=$W/$TAG-smc.log SCLK_GUARD=1 \
  nohup $B/clamp-watchdog-v2.sh >> $W/$TAG-watchdog.out 2>&1 &
nohup $B/smc-log.sh $W/$TAG-smc.log >/dev/null 2>&1 &
sleep 6
log "=== start: $GPU_CAP W/die, build $(basename $P) $($P/bin/llama-batched-bench --version 2>&1 | grep -oE 'build [0-9]+ commit [0-9a-f]+' | head -1), ggml $(sha256sum $P/lib/libggml-hip.so.0 | cut -c1-16)"
log "cells: $CELLS"

{ echo "# $TAG — capacity ladder, q8_0 KV, $GPU_CAP W/die   $(date -Is)"
  echo
  echo "Build \`$P\` (build 11067, commit 1d1361e7a) on ROCm 10.0, tp4 tensor split, \`-fa on\`,"
  echo "\`-ngl all\`, \`-b 2048 -ub 2048\`, \`-ntg 128\`, gfx906.env (custom AR + corrected XGMI ring)."
  echo "Model Qwen3.8-27B-**Q8_0** (model quant, R2.5) — distinct from the q8_0 **KV** type."
  echo "\`$(state_line)\`"
  echo
  echo "R3.1 floor is 12 tok/s **per request**. In batched-bench every sequence decodes in"
  echo "lockstep, so per-slot = aggregate/slots exactly — there is no median/p10 spread here."
  echo
  echo "| slots | depth | fits | pp t/s | tg t/s agg | **per slot** | R3.1 | KV MiB/die | compute MiB/die | peak VRAM GiB/die | cell s |"
  echo "|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|"; } > $OUT

peak_vram() {  # background: track peak VRAM per die into $1
  local f=$1; : > $f
  ( declare -A pk; while true; do
      for d in $DIES; do
        v=$(cat /sys/bus/pci/devices/0000:$d:00.0/mem_info_vram_used 2>/dev/null || echo 0)
        [ "${v:-0}" -gt "${pk[$d]:-0}" ] && pk[$d]=$v
      done
      : > $f.tmp; for d in $DIES; do echo "$d ${pk[$d]:-0}" >> $f.tmp; done; mv $f.tmp $f
      sleep 3
    done ) & VS=$!
}

cell() {   # cell SLOTS DEPTH ; never ends in sleep (trap: a run() ending in sleep always returns 0)
  local slots=$1 depth=$2 kv=q8_0
  local name="$slots x $((depth/1024))K"
  local base=$LOGD/$kv-$slots-$depth
  local ctx=$(( slots * (depth + 256) ))
  log "--- cell $name : ctx $ctx, npp $depth, npl $slots"
  peak_vram $base.vram
  local t0=$(date +%s)
  timeout 5400 $P/bin/llama-batched-bench -m $M $D4 -fa on -ctk $kv -ctv $kv \
      -b 2048 -ub 2048 -c $ctx -npp $depth -ntg 128 -npl $slots \
      > $base.md 2>$base.log
  local rc=$? t1=$(date +%s); local secs=$(( t1 - t0 ))
  kpid "${VS:-}"; VS=""
  local pv=$(awk '{printf "%.1f ", $2/1073741824}' $base.vram 2>/dev/null)
  # offload assertion: a partially-offloaded cell is not a valid measurement
  local offl=$(grep -oE 'offloaded [0-9]+/[0-9]+ layers' $base.log | head -1)
  local kvd=$(grep -oE 'ROCm0 KV buffer size = *[0-9.]+ MiB' $base.log | head -1 | grep -oE '[0-9.]+' | head -1)
  local cbd=$(grep -oE 'ROCm0 compute buffer size = *[0-9.]+ MiB' $base.log | head -1 | grep -oE '[0-9.]+' | head -1)
  local row=$(grep -E "^\| *$depth " $base.md | tail -1)
  if [ -n "$row" ]; then
    local pp tg per gate
    pp=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$7); print $7}')
    tg=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$9); print $9}')
    per=$(awk -v t="$tg" -v s=$slots 'BEGIN{printf "%.2f", t/s}')
    gate=$(awk -v p="$per" 'BEGIN{print (p>=12)?"**PASS**":"FAIL"}')
    echo "| $slots | $((depth/1024))K | yes | $pp | $tg | **$per** | $gate | ${kvd:-?} | ${cbd:-?} | ${pv:-?} | $secs |" >> $OUT
    log "cell $name: pp $pp, tg $tg agg, $per per slot -> $(echo $gate|tr -d '*'), $offl, peak VRAM ${pv}GiB, ${secs}s"
  else
    local why=$(grep -m1 -oiE 'out of memory|failed to allocate[^\n]{0,40}|hipErrorOutOfMemory|unable to allocate[^\n]{0,40}|error[^\n]{0,60}' $base.log | head -1)
    [ $rc -eq 124 ] && why="timeout 5400s"
    echo "| $slots | $((depth/1024))K | **no** | — | — | — | — | ${kvd:-?} | ${cbd:-?} | ${pv:-?} | $secs |" >> $OUT
    echo "|  |  | _rc=$rc: ${why:-no result row}_ |  |  |  |  |  |  |  |  |" >> $OUT
    log "cell $name: DOES NOT FIT / FAILED rc=$rc ($why), peak VRAM ${pv}GiB, ${secs}s"
  fi
  [ -n "$offl" ] && echo "$name: $offl" >> $W/$TAG-offload.txt
  return 0
}

for c in $CELLS; do IFS=: read s d <<< "$c"; cell $s $d; done

{ echo; echo "Peak SMC DC total during the ladder: $(sed -n 's/.*PZ0G=\([0-9]*\).*/\1/p' $W/$TAG-smc.log 2>/dev/null | sort -n | tail -1) W (envelope 1228 W)."
  echo; echo "Layer offload per cell (all must read all-on-GPU):"; sed 's/^/  - /' $W/$TAG-offload.txt 2>/dev/null
  echo; echo "# done $(date -Is)"; } >> $OUT

trap - INT TERM EXIT
kpid "${SAMP:-}"
for d in $DIES; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore
touch $DONEFLAG
log "=== ALLDONE"
