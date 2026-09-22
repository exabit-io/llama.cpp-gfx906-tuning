#!/bin/bash
# gate12.sh — Part 6 gates 1 and 2, on the instrument the bundle result was measured with.
#
# GATE 1 (zero point): D2 defines the zero point as STOCK upstream v0.4.1. Every baseline so far
# was on the substrate build, so T = B + R + sum(Pi) had no fixed B. Measured here at the same two
# cells, cap, env and n as cmp-terms.sh, so B, R and R+ours are directly comparable numbers.
#
# GATE 2 (single-user axis): D1 requires a per-axis A/B, and until "single-stream decode" is an
# OPERATIONAL instrument with a v0.4.1/ROCm-10.0 baseline, every patch can only be binned
# multi-user -- gfx906-both and gfx906-single are unreachable by construction. Definition adopted:
#
#   single-user instrument = llama-batched-bench -npl 1 -npp 32768 -ntg 1024, four dies, -sm tensor,
#   q8_0 KV, 125 W, n=4; metrics: decode tok/s (one stream) and prefill t/s.
#
# Four dies and one stream, because that is what the single-user profile actually runs; 32K because
# of the context floor rule (2K cells are never a gate). Three arms so the axis gets its zero point
# (stock), its baseline (substrate) and the bundle in one job.
set -u
W=/root/night-20260919; M=/root/models/Qwen3.8-27B-Q8_0.gguf; R=/root/rocm-tests/bench
S=/root/build-stock-v041          # B : stock upstream v0.4.1, the zero point
A=/root/build-substrate-v041      # R : substrate
C=/root/build-c4series            # R + our 20 terms
log(){ echo "$(date -Is) [g12] $*" | tee -a $W/gate12.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
# Refuse to produce a zero point from a build that is not there. Silent-skip is the failure mode
# that made two gates do nothing last night.
for b in $S $A $C; do
  [ -x "$b/bin/llama-batched-bench" ] || { echo "FATAL: $b/bin/llama-batched-bench missing — gate cannot run" >&2; exit 1; }
done
. $R/gpu-test-env.sh
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/gate12-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE="^/bin/bash $W/gate12[.]sh" DONEFLAG=$W/.gate12-done SMCLOG=$W/gate12-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/gate12-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/gate12-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/gate12.tsv
cell(){ # cell ARM BUILD SLOTS DEPTH REP
  local arm=$1; local bld=$2; local s=$3; local d=$4; local rep=$5
  local out=$W/g12-$arm-$s-$d-$rep.md
  LD_LIBRARY_PATH=$bld/bin:$bld/lib:/opt/rocm/core-10.0/lib timeout 3600 $bld/bin/llama-batched-bench \
    -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -ctk q8_0 -ctv q8_0 \
    -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s > $out 2>$W/g12-$arm-$s-$d-$rep.log
  local row per pp
  row=$(grep -E "^\| *$d " $out | tail -1)
  # T5b: an empty row means the CELL FAILED. Logging NA and moving on would let the analysis drop it
  # silently and compute a zero point from fewer reps than it claims.
  if [ -z "${row:-}" ]; then
    log "CELL FAILED: $arm ${s}x$((d/1024))K rep$rep produced no result row — last stderr:"
    tail -3 $W/g12-$arm-$s-$d-$rep.log 2>/dev/null | sed 's/^/      /' | tee -a $W/gate12.progress
    printf "%s\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$((s))x$((d/1024))K" "$rep" >> $W/gate12.tsv
    return 0
  fi
  pp=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$7); print $7}')
  per=$(echo "$row" | awk -F'|' -v s=$s '{gsub(/ /,"",$9); printf "%.4f", $9/s}')
  printf "%s\t%s\t%s\t%s\t%s\n" "$arm" "$((s))x$((d/1024))K" "$rep" "${per:-NA}" "${pp:-NA}" >> $W/gate12.tsv
  log "$arm ${s}x$((d/1024))K rep$rep: decode ${per:-NA} tok/s/slot, prefill ${pp:-NA} t/s"
}
log "=== GATE 1: stock v0.4.1 zero point at the pinned recipe, 4x32K and 4x64K, n=4"
for rep in 1 2 3 4; do cell B $S 4 32768 $rep; cell B $S 4 65536 $rep; done
log "=== GATE 2: single-user axis instrument, 1 stream x 32K, three arms, n=4, order rotated"
for rep in 1 2 3 4; do
  case $((rep % 3)) in
    1) cell Bsu $S 1 32768 $rep; cell Rsu $A 1 32768 $rep; cell RoursSU $C 1 32768 $rep;;
    2) cell Rsu $A 1 32768 $rep; cell RoursSU $C 1 32768 $rep; cell Bsu $S 1 32768 $rep;;
    0) cell RoursSU $C 1 32768 $rep; cell Bsu $S 1 32768 $rep; cell Rsu $A 1 32768 $rep;;
  esac
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.gate12-done
log "=== GATE12 DONE ==="
