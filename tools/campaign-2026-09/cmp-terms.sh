#!/bin/bash
# First controlled comparison: substrate-only (R) vs substrate + our 20 terms (R + ours-bundle).
# Same cells, same cap, same env — one variable set. n=4, interleaved A/B/B/A so drift cancels
# (Part 1 s1.10). This is the bundle arm; the survey then decomposes which terms carry it.
set -u
W=/root/night-20260919; M=/root/models/Qwen3.8-27B-Q8_0.gguf; R=/root/rocm-tests/bench
A=/root/build-substrate-v041      # R
B=/root/build-c4series            # R + our 20 terms
log(){ echo "$(date -Is) [cmp] $*" | tee -a $W/cmp.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
. $R/gpu-test-env.sh
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/cmp-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE="^/bin/bash $W/cmp-terms[.]sh" DONEFLAG=$W/.cmp-done SMCLOG=$W/cmp-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/cmp-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/cmp-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/cmp.tsv
cell(){ # cell ARM BUILD SLOTS DEPTH REP
  local arm=$1; local bld=$2; local s=$3; local d=$4; local rep=$5
  local out=$W/cmp-$arm-$s-$d-$rep.md
  LD_LIBRARY_PATH=$bld/bin:$bld/lib:/opt/rocm/core-10.0/lib timeout 3600 $bld/bin/llama-batched-bench \
    -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -ctk f16 -ctv f16   `# production KV (lead 2026-09-23); q8_0 cells are superseded` \
    -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s > $out 2>$W/cmp-$arm-$s-$d-$rep.log
  local row per pp
  row=$(grep -E "^\| *$d " $out | tail -1)
  pp=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$7); print $7}')
  per=$(echo "$row" | awk -F'|' -v s=$s '{gsub(/ /,"",$9); printf "%.4f", $9/s}')
  printf "%s\t%s\t%s\t%s\t%s\n" "$arm" "$((s))x$((d/1024))K" "$rep" "${per:-NA}" "${pp:-NA}" >> $W/cmp.tsv
  log "$arm ${s}x$((d/1024))K rep$rep: decode ${per:-NA} tok/s/slot, prefill ${pp:-NA} t/s"
}
log "=== start: R (substrate) vs R+ours (20 terms), n=4, ABBA interleaved"
for rep in 1 2 3 4; do
  if [ $((rep % 2)) -eq 1 ]; then o1=R; b1=$A; o2=Rours; b2=$B; else o1=Rours; b1=$B; o2=R; b2=$A; fi
  cell $o1 $b1 4 32768 $rep; cell $o2 $b2 4 32768 $rep
  cell $o1 $b1 4 65536 $rep; cell $o2 $b2 4 65536 $rep
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.cmp-done; log "=== ALLDONE"
