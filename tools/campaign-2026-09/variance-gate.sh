#!/bin/bash
# Part 6 GATE 3: run-to-run variance of the harness. Sets n, and therefore the whole budget:
#   sigma 0.3-0.5% -> n=2 detects a 2% effect;  sigma 2% -> n=16 and the campaign quadruples.
#
# Design: INTERLEAVE the two cell types rather than running each back-to-back. Back-to-back runs
# share thermal and cache state, which UNDERSTATES sigma; interleaving also exposes drift, and the
# campaign's own runs will be blocked and interleaved (Part 1 s1.10), so this matches them.
#   proxy cell      4 x 32K, -ntg 1024   (the screening metric, ~2.8 min)
#   design-pt cell  4 x 64K, -ntg 1024   (the confirmation metric, ~5.6 min)
set -u
W=/root/night-20260919; B=/root/build-substrate-v041; M=/root/models/Qwen3.8-27B-Q8_0.gguf
R=/root/rocm-tests/bench
log() { echo "$(date -Is) [variance] $*" | tee -a $W/variance.progress; }
. $R/gpu-test-env.sh
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
export LD_LIBRARY_PATH=$B/bin:$B/lib:/opt/rocm/core-10.0/lib
kpid() { [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/variance-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE="^/bin/bash $W/variance-gate[.]sh" DONEFLAG=$W/.variance-done SMCLOG=$W/variance-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/variance-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/variance-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/variance.tsv
cell() { # cell TAG SLOTS DEPTH REP
  # NB separate `local` statements: bash expands EVERY word of a single `local a=$1 b=$a` before
  # performing any assignment, so a self-referencing one leaves the later vars empty — and fatal
  # under `set -u`. This exact bug killed fanpower.sh first; see trap T10.
  local tag=$1; local s=$2; local d=$3; local rep=$4
  local out=$W/variance-$tag-$rep.md
  timeout 3600 $B/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16   `# production KV (lead 2026-09-23); q8_0 cells are superseded` -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s \
    > $out 2>$W/variance-$tag-$rep.log
  local row per pp
  row=$(grep -E "^\| *$d " $out | tail -1)
  pp=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$7); print $7}')
  per=$(echo "$row" | awk -F'|' -v s=$s '{gsub(/ /,"",$9); printf "%.4f", $9/s}')
  printf "%s\t%s\t%s\t%s\n" "$tag" "$rep" "${per:-NA}" "${pp:-NA}" >> $W/variance.tsv
  log "$tag rep$rep: per-slot decode ${per:-NA} tok/s, prefill ${pp:-NA} t/s"
}
log "=== start: interleaved proxy(4x32K) and design(4x64K) cells, 125 W, -ntg 1024"
for rep in 1 2 3 4 5 6; do
  cell proxy 4 32768 $rep
  [ $rep -le 4 ] && cell design 4 65536 $rep
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.variance-done; log "=== ALLDONE"
