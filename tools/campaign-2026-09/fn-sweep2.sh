#!/bin/bash
# fn-sweep.sh — Flash-Next offload sweep at the 64K FLOOR.
#
# Lead, 2026-09-23: "64k is absolute floor." The earlier load test used 2K, which was fine as a load
# test and worthless as a measurement -- and I then quoted its decode numbers and drew an R3.1
# conclusion from them. A thinking model can spend 2K talking to itself before it answers; nothing at
# that depth represents this service. The floor was 32K in REQUIREMENTS R2.2 and is now 64K.
#
# The question: where is the knee between VRAM freed and decode lost? At 2K the extremes were
#   plain      418 t/s prefill, 32.75 tok/s decode, 28.5 GiB/die
#   ncmoe 41    98 t/s prefill, 10.39 tok/s decode, 12.9 GiB/die
# and both of those figures are now discarded. This measures offload 0, 12, 24, 32, 41 at 1x64K, which
# is the floor, and reports VRAM per die alongside so the trade is visible rather than inferred.
set -u
W=/root/night-20260919; B=/root/build-faq-allquants; R=/root/rocm-tests/bench
MD=/root/models/Qwen3.8-Flash-Next-UD-Q4_K_XL
M=$MD/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
log(){ echo "$(date -Is) [fnsw2] $*" | tee -a $W/fnsweep2.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
vram(){ local t=0; for d in 0b 0e 1b 1e; do t=$((t+$(cat /sys/bus/pci/devices/0000:$d:00.0/mem_info_vram_used 2>/dev/null || echo 0))); done; echo $t; }
rm -f $W/.fnsweep2-done
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml GGML_CUDA_ALLREDUCE=nccl
export GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
export LLAMA_PLE_SHARD=1
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash /root/rocm-tests/bench/clamp-watchdog-v2[.]sh" 2>/dev/null
  pkill -f "^/bin/bash /root/rocm-tests/bench/smc-log[.]sh" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/fnsweep2-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE_PID=$$ DONEFLAG=$W/.fnsweep2-done SMCLOG=$W/fnsweep2-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/fnsweep2-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/fnsweep2-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/fnsweep2.tsv
cell(){ # cell NCMOE SLOTS DEPTH REP
  local nc=$1 s=$2 d=$3 rep=$4
  local extra=(); [ "$nc" -gt 0 ] && extra=(--n-cpu-moe "$nc")
  local out=$W/fs2-$nc-$s-$d-$rep.md
  LD_LIBRARY_PATH=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib timeout 5400 \
    $B/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -lm mlock \
    -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 256 -npl $s \
    "${extra[@]}" > $out 2>$W/fs2-$nc-$s-$d-$rep.log &
  local bp=$!; local peak=0
  while kill -0 $bp 2>/dev/null; do v=$(vram); [ "$v" -gt "$peak" ] && peak=$v; sleep 4; done
  wait $bp 2>/dev/null
  local row; row=$(grep -E "^\| *$d " $out | tail -1)
  if [ -z "$row" ]; then
    log "FAILED: ncmoe=$nc ${s}x$((d/1024))K — first errors:"
    grep -m3 -iE 'error|failed|out of memory|abort' $W/fs2-$nc-$s-$d-$rep.log | sed 's/^/      /' | tee -a $W/fnsweep2.progress
    printf "%s\t%sx%sK\t%s\tFAILED\tFAILED\tFAILED\n" "$nc" "$s" "$((d/1024))" "$rep" >> $W/fnsweep2.tsv; return 0
  fi
  local pp dec
  pp=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$7); print $7}')
  dec=$(echo "$row" | awk -F'|' -v s=$s '{gsub(/ /,"",$9); printf "%.4f", $9/s}')
  printf "%s\t%sx%sK\t%s\t%s\t%s\t%.1f\n" "$nc" "$s" "$((d/1024))" "$rep" "$dec" "$pp" "$(echo "$peak/4/1073741824" | bc -l)" >> $W/fnsweep2.tsv
  log "ncmoe=$nc ${s}x$((d/1024))K rep$rep: decode ${dec} tok/s/slot, prefill ${pp} t/s, peak VRAM $(echo "scale=1;$peak/4/1073741824" | bc)GiB/die"
}
log "=== Flash-Next: offload x SLOTS at the 64K floor. 1 slot found ncmoe=12 the knee (14.29 tok/s,"
log "=== 27.6 GiB/die) and ncmoe=0 does not fit. KV grows with slots, so the viable level must rise."
# Bracket each slot count: the lowest offload that fits, and enough above it to find the R3.1 boundary.
for nc in 8 12 16 24; do cell $nc 2 65536 1; done
for nc in 16 24 32 41; do cell $nc 4 65536 1; done
for nc in 32 41 48; do cell $nc 8 65536 1; done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash /root/rocm-tests/bench/clamp-watchdog-v2[.]sh" 2>/dev/null
pkill -f "^/bin/bash /root/rocm-tests/bench/smc-log[.]sh" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.fnsweep2-done; log "=== FN SWEEP DONE ==="
