#!/bin/bash
# fn-load.sh — get Qwen3.8-Flash-Next running in the PRODUCTION configuration.
#
# Lead, 2026-09-23: production is Flash-Next with `--n-cpu-moe 41` and `--no-mmap --mlock`. In this
# llama.cpp those last two are one option: `-lm mlock` (--load-mode auto|mmap|mlock|mmap+mlock).
# An earlier attempt failed instantly on `--no-mmap` -- my error, not the model's.
#
# This is a LOAD test first and a measurement second. The 2026-09-07/08 binaries could not load this
# model at all, and whether the v0.4.1 substrate can is unknown. Each step reports what actually
# happened rather than only whether it exited 0.
set -u
W=/root/night-20260919; B=/root/build-faq-allquants; R=/root/rocm-tests/bench
MD=/root/models/Qwen3.8-Flash-Next-UD-Q4_K_XL
M=$MD/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
log(){ echo "$(date -Is) [fnload] $*" | tee -a $W/fnload.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
[ -f "$M" ] || { echo "FATAL: $M missing" >&2; exit 1; }
rm -f $W/.fnload-done
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
setmax; start_sampler $W/fnload-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE_PID=$$ DONEFLAG=$W/.fnload-done SMCLOG=$W/fnload-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/fnload-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/fnload-smc.log >/dev/null 2>&1 &
sleep 6
try(){ # try TAG extra-args...
  local tag=$1; shift
  local out=$W/fnl-$tag.md; local err=$W/fnl-$tag.log
  log "attempt '$tag': $*"
  timeout 3600 env LD_LIBRARY_PATH=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib \
    $B/bin/llama-batched-bench -m $M -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 \
    -c 8192 -npp 2048 -ntg 32 -npl 1 "$@" > $out 2>$err
  local rc=$?
  local row; row=$(grep -E '^\| *2048 ' $out | tail -1)
  if [ -n "$row" ]; then
    log "  OK  $(echo "$row" | tr -s ' ' | cut -c1-92)"
    grep -m1 -iE 'n_expert|expert_used' $err | sed 's/^/        /' | tee -a $W/fnload.progress
    return 0
  fi
  log "  FAILED exit=$rc — what it said:"
  grep -m5 -iE 'error|failed|unsupported|unknown|abort|assert|out of memory|not supported|cannot' $err \
    | sed 's/^/        /' | tee -a $W/fnload.progress
  return 1
}
# escalating: does it load at all, then add the production flags one at a time so a failure is attributable
try plain        --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all
try lm-mlock     --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -lm mlock
try ncmoe41      --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -lm mlock --n-cpu-moe 41
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash /root/rocm-tests/bench/clamp-watchdog-v2[.]sh" 2>/dev/null
pkill -f "^/bin/bash /root/rocm-tests/bench/smc-log[.]sh" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.fnload-done; log "=== FN LOAD TEST DONE ==="
