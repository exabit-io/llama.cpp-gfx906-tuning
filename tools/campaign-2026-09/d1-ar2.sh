#!/bin/bash
# d1-ar2.sh — reps 5-6: FRESH data to test custom AR under the CORRECTED statistical rule.
#
# Why this run exists, stated plainly: reps 1-4 showed +9.97% multi decode and +25.95% single decode
# with completely non-overlapping distributions, and BOTH failed the pre-registered test. Two flaws in
# MY method, not in the data:
#   1. the MEDIAN estimator adopted for the single-user axis on 2026-09-20 cannot reach q<0.10 at n=4 --
#      the median of 4 is the mean of the middle two, so a 3-1 split ties the perfect 4-0 split and ten
#      of 70 arrangements match the observed statistic. Robustness to the stall ate the power.
#   2. BH over 4 tests (decode AND prefill on each axis) needs p<=0.025 at rank 1, and two of the four
#      were near-certain nulls. Putting expected nulls in the FDR family inflates the correction.
# Corrected rule: PRIMARY endpoint = decode (the claim being made); prefill is a secondary safety check
# outside the FDR family; the single-user test is RANK-SUM, which keeps power and still survives a stall.
#
# Changing the analysis after seeing the data is p-hacking unless it is verified on new data. That is
# what these reps are for. Reps 1-4 are reported as the screen that motivated the fix, not as evidence.
#
# WHY THIS IS STILL OPEN after the confirmation. The gate is confirmed to improve prefill on both axes
# (+23.8% multi, +28.4% single). Custom AR ungated was screened at +8.8% decode / -19.0% prefill. But
# the comparison production actually needs is (AR OFF) vs (AR ON + gate), and no run has made it on ONE
# build: the AR-off arm was measured on the substrate, which does not compile in the gate knob at all,
# so that contrast crossed two binaries and is not a clean term.
#
# Here both arms are the BUNDLE build, which has both knobs, so exactly one variable moves:
#   arm AR-off       GGML_ENABLE_CUSTOM_AR=0                              (no custom AR, no gate)
#   arm AR-on-gated  GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481      (custom AR, gated)
#
# If AR-on-gated does not beat AR-off, then custom AllReduce plus its gate is 1020 lines of kernel that
# buys nothing, and the pair is bin regresses-both or neutral-drop rather than required substrate.
# n=4, both axes, median on single-user per gate 2's amendment. setsid at launch.
set -u
W=/root/night-20260919; M=/root/models/Qwen3.8-27B-Q8_0.gguf; R=/root/rocm-tests/bench
A=/root/build-substrate-v041; C=/root/build-c4series
log(){ echo "$(date -Is) [ar2] $*" | tee -a $W/d1ar2.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
for b in $A $C; do [ -x "$b/bin/llama-batched-bench" ] || { echo "FATAL: $b missing" >&2; exit 1; }; done
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
export HSA_FORCE_FINE_GRAIN_PCIE=1
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/d1ar2-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
rm -f $W/.d1ar2-done
QUEUE_PID=$$ DONEFLAG=$W/.d1ar2-done SMCLOG=$W/d1ar2-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/d1ar2-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/d1ar2-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/d1ar2.tsv
cell(){ # cell ARM BUILD SLOTS DEPTH REP EXTRA...
  local arm=$1; local bld=$2; local s=$3; local d=$4; local rep=$5; shift 5
  local out=$W/d2-$arm-$s-$d-$rep.md
  timeout 3600 env LD_LIBRARY_PATH=$bld/bin:$bld/lib:/opt/rocm/core-10.0/lib \
    $bld/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16   `# production KV (lead 2026-09-23); q8_0 cells are superseded` -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s "$@" \
    > $out 2>$W/d2-$arm-$s-$d-$rep.log
  local row per pp
  row=$(grep -E "^\| *$d " $out | tail -1)
  if [ -z "${row:-}" ]; then
    log "CELL FAILED: $arm ${s}x$((d/1024))K rep$rep"; tail -3 $W/d2-$arm-$s-$d-$rep.log | sed 's/^/      /' | tee -a $W/d1ar2.progress
    printf "%s\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$((s))x$((d/1024))K" "$rep" >> $W/d1ar2.tsv; return 0
  fi
  pp=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$7); print $7}')
  per=$(echo "$row" | awk -F'|' -v s=$s '{gsub(/ /,"",$9); printf "%.4f", $9/s}')
  printf "%s\t%s\t%s\t%s\t%s\n" "$arm" "$((s))x$((d/1024))K" "$rep" "${per:-NA}" "${pp:-NA}" >> $W/d1ar2.tsv
  log "$arm ${s}x$((d/1024))K rep$rep: decode ${per:-NA} tok/s/slot, prefill ${pp:-NA} t/s"
}
log "=== does custom AR earn its place? both arms on the BUNDLE build, one variable"
for rep in 5 6; do
  if [ $((rep % 2)) -eq 1 ]; then
    GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 cell M-AR-on-gated $C 4 65536 $rep
    GGML_ENABLE_CUSTOM_AR=0                          cell M-AR-off     $C 4 65536 $rep
  else
    GGML_ENABLE_CUSTOM_AR=0                          cell M-AR-off     $C 4 65536 $rep
    GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 cell M-AR-on-gated $C 4 65536 $rep
  fi
done
for rep in 5 6; do
  if [ $((rep % 2)) -eq 1 ]; then
    GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 cell S-AR-on-gated $C 1 32768 $rep
    GGML_ENABLE_CUSTOM_AR=0                          cell S-AR-off     $C 1 32768 $rep
  else
    GGML_ENABLE_CUSTOM_AR=0                          cell S-AR-off     $C 1 32768 $rep
    GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 cell S-AR-on-gated $C 1 32768 $rep
  fi
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.d1ar2-done
log "=== AR-EARNS-ITS-PLACE DONE ==="
