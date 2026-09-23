#!/bin/bash
# flashnext.sh — measure the ACTUAL production configuration, which is not what this campaign has been
# optimising.
#
# Lead, 2026-09-23: production is Qwen3.8-Flash-Next with `--n-cpu-moe 41 --no-mmap --mlock`, not
# Qwen3.8-27B. That changes the bottleneck entirely: 41 layers of MoE expert weights live in host RAM
# and stream over PCIe, so this is a hybrid CPU/GPU workload, not an HBM-bandwidth one. None of the 27B
# KV findings can be assumed to transfer -- KV is 3.1% of per-step bytes on this model against 31.5% on
# the 27B, and the expert traffic is a new term that does not exist there at all.
#
# Minimal decisive set: does it load, and does the KV type matter here at all.
#   arms  f16-f16 vs q8_0-q8_0      (the only two that matter until we know the shape)
#   cells 4x32K and 1x32K           (start small: expert streaming may dominate everything)
set -u
W=/root/night-20260919; B=/root/build-faq-allquants; R=/root/rocm-tests/bench
M=/root/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
CM=/root/llama.cpp-benchmarking/tools/cell-metrics.py
NF=/root/llama.cpp-benchmarking/tools/assert-no-fa-fallback.sh
log(){ echo "$(date -Is) [fn] $*" | tee -a $W/flashnext.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
[ -f "$M" ] || { echo "FATAL: $M missing" >&2; exit 1; }
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml GGML_CUDA_ALLREDUCE=nccl
export GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
export LLAMA_PLE_SHARD=1     # the fork's PLE gather-table sharding; the 27B does not need it
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash /root/rocm-tests/bench/clamp-watchdog-v2[.]sh" 2>/dev/null
  pkill -f "^/bin/bash /root/rocm-tests/bench/smc-log[.]sh" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/flashnext-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
rm -f $W/.flashnext-done
QUEUE_PID=$$ DONEFLAG=$W/.flashnext-done SMCLOG=$W/flashnext-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/flashnext-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/flashnext-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/flashnext.tsv
cell(){ # cell KV SLOTS DEPTH REP
  local kv=$1 s=$2 d=$3 rep=$4
  local out=$W/fn-$kv-$s-$d-$rep.md
  log "starting $kv ${s}x$((d/1024))K rep$rep (103.7 GiB model, --no-mmap --mlock: load takes minutes)"
  LD_LIBRARY_PATH=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib timeout 10800 \
    $B/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    --n-cpu-moe 41 --no-mmap --mlock \
    -fa on -ctk $kv -ctv $kv -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 256 -npl $s \
    > $out 2>$W/fn-$kv-$s-$d-$rep.log
  local rc=$?
  if [ $rc -ne 0 ] && ! grep -qE '^\| *'"$d"' ' $out 2>/dev/null; then
    log "FAILED (exit $rc): $kv ${s}x$((d/1024))K — first errors:"
    grep -m4 -iE 'error|failed|abort|out of memory|unsupported|unknown' $W/fn-$kv-$s-$d-$rep.log | sed 's/^/      /' | tee -a $W/flashnext.progress
    printf "%s\t%sx%sK\t%s\tFAILED\tFAILED\n" "$kv" "$s" "$((d/1024))" "$rep" >> $W/flashnext.tsv; return 0
  fi
  $NF "$W/fn-$kv-$s-$d-$rep.log" >/dev/null 2>&1 || log "WARNING: FA fallback in $kv ${s}x$((d/1024))K"
  local row per pp
  row=$(grep -E "^\| *$d " $out | tail -1)
  pp=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$6); print $6}')
  per=$(echo "$row" | awk -F'|' '{gsub(/ /,"",$8); print $8}')
  printf "%s\t%sx%sK\t%s\t%s\t%s\n" "$kv" "$s" "$((d/1024))" "$rep" "${per:-NA}" "${pp:-NA}" >> $W/flashnext.tsv
  log "$kv ${s}x$((d/1024))K rep$rep: T_TG ${per:-NA}s, T_PP ${pp:-NA}s | $(echo "$row" | tr -s ' ' | cut -c1-90)"
}
log "=== Flash-Next, production flags: --n-cpu-moe 41 --no-mmap --mlock, 4 dies tensor-split"
cell f16 1 32768 1
cell q8_0 1 32768 1
cell f16 4 32768 1
cell q8_0 4 32768 1
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash /root/rocm-tests/bench/clamp-watchdog-v2[.]sh" 2>/dev/null
pkill -f "^/bin/bash /root/rocm-tests/bench/smc-log[.]sh" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.flashnext-done; log "=== FLASH-NEXT DONE ==="
