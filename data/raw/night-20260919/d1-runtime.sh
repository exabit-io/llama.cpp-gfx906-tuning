#!/bin/bash
# d1-runtime.sh — D-1 screen for the candidates that are RUNTIME-GATED, so they need no build surgery.
#
# Why these three first: the substrate is a SQUASH merge, so an individual mxxm commit cannot be left
# out of it, and mxxm's own commits are expressed against b10760, not v0.4.1 — they are not composable
# terms. But three of the eight D-1 candidates have runtime switches compiled into builds we already
# have, which makes them measurable TODAY as clean single-variable contrasts:
#
#   E1  custom allreduce          substrate, GGML_ENABLE_CUSTOM_AR=1 vs 0      (candidate 3)
#   E2  q8 weight repack          substrate, default vs --no-repack            (candidate 6)
#   E3  the AR size gate itself   bundle, GGML_TP_AR_MAX_NE=20481 vs unset     (terms/06)
#
# E3 exists because the knob is compiled into the bundle build but NOT into the substrate build, so
# every run today set an env var that only one arm could honour. That is correct attribution for the
# bundle (terms/06 IS the knob), but it means env has to be controlled per arm from here on, never
# inherited from gfx906.env. This script sets every knob explicitly and unsets what it is not testing.
#
# n=2 at the pinned design point (4x64K) = the D-4 SCREEN. Screen data never becomes confirmation data
# (winner's curse); anything that screens interesting is re-measured fresh at n>=4 on both axes.
set -u
W=/root/night-20260919; M=/root/models/Qwen3.8-27B-Q8_0.gguf; R=/root/rocm-tests/bench
A=/root/build-substrate-v041; C=/root/build-c4series
log(){ echo "$(date -Is) [d1] $*" | tee -a $W/d1.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
. $R/gpu-test-env.sh
# NOTE: gfx906.env is deliberately NOT sourced. Each arm below states its own knobs.
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
export HSA_FORCE_FINE_GRAIN_PCIE=1
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/d1-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
rm -f $W/.d1-done
QUEUE_PID=$$ DONEFLAG=$W/.d1-done SMCLOG=$W/d1-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/d1-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/d1-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/d1.tsv
cell(){ # cell ARM BUILD SLOTS DEPTH REP EXTRA_FLAGS...
  local arm=$1; local bld=$2; local s=$3; local d=$4; local rep=$5; shift 5
  local out=$W/d1-$arm-$s-$d-$rep.md
  timeout 3600 env LD_LIBRARY_PATH=$bld/bin:$bld/lib:/opt/rocm/core-10.0/lib \
    $bld/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16   `# production KV (lead 2026-09-23); q8_0 cells are superseded` -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s "$@" \
    > $out 2>$W/d1-$arm-$s-$d-$rep.log
  local row per pp
  row=$(grep -E "^\| *$d " $out | tail -1)
  if [ -z "${row:-}" ]; then
    log "CELL FAILED: $arm ${s}x$((d/1024))K rep$rep — last stderr:"
    tail -3 $W/d1-$arm-$s-$d-$rep.log 2>/dev/null | sed 's/^/      /' | tee -a $W/d1.progress
    printf "%s\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$((s))x$((d/1024))K" "$rep" >> $W/d1.tsv; return 0
  fi
  pp=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$7); print $7}')
  per=$(echo "$row" | awk -F'|' -v s=$s '{gsub(/ /,"",$9); printf "%.4f", $9/s}')
  printf "%s\t%s\t%s\t%s\t%s\n" "$arm" "$((s))x$((d/1024))K" "$rep" "${per:-NA}" "${pp:-NA}" >> $W/d1.tsv
  log "$arm ${s}x$((d/1024))K rep$rep: decode ${per:-NA} tok/s/slot, prefill ${pp:-NA} t/s"
}
log "=== D-1 runtime screen, n=2 at the 4x64K design point, 125 W, env stated per arm"
for rep in 1 2; do
  # E1: custom allreduce, on the substrate. The MAX_NE gate knob is NOT in this build, so AR is ungated.
  GGML_ENABLE_CUSTOM_AR=1 cell E1-AR-on   $A 4 65536 $rep
  GGML_ENABLE_CUSTOM_AR=0 cell E1-AR-off  $A 4 65536 $rep
  # E2: q8 weight repack, on the substrate (default is enabled).
  GGML_ENABLE_CUSTOM_AR=1 cell E2-repack-on  $A 4 65536 $rep
  GGML_ENABLE_CUSTOM_AR=1 cell E2-repack-off $A 4 65536 $rep --no-repack
  # E3: does the AR size gate matter? Only the bundle build has the knob.
  GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 cell E3-gate-on  $C 4 65536 $rep
  GGML_ENABLE_CUSTOM_AR=1                          cell E3-gate-off $C 4 65536 $rep
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.d1-done
log "=== D1 RUNTIME SCREEN DONE ==="
