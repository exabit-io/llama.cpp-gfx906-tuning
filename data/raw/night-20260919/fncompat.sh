#!/bin/bash
# fncompat.sh — Flash-Next (qwen4exp) COMPATIBILITY gate for every binrun patchset build. Pass/fail only;
# no performance number from this script is a result (lead 2026-09-24: the only purpose of Flash-Next here
# is to confirm each patchset we classify is compatible with the qwen4exp architecture).
#
# Per build: production flags (--n-cpu-moe 41, mlock, LLAMA_PLE_SHARD=1), four dies -sm tensor, f16 KV,
# greedy 64-token completion of a fixed prompt. PASS = exits 0 and generates text. The text is compared
# with the base build's: identical, or different (then read it — kernel changes may legitimately
# reorder float sums, but garbage means the patchset breaks qwen4exp).
# Waits for binrun.sh by PID. Host CPU runs the offloaded experts, so the 150 W RAPL cap (setmax) is mandatory.
set -u
W=/root/night-20260919; R=/root/rocm-tests/bench
M=/root/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
# ROUND / ARMS_FILE (env) as in binrun.sh; empty = round 1's names and arms.
ROUND=${ROUND:-}; SFX=${ROUND:+-$ROUND}; ARMS_FILE=${ARMS_FILE:-$W/binrun-arms.txt}
P=$W/fncompat$SFX.progress; TSV=$W/fncompat$SFX.tsv; DONE=$W/.fncompat$SFX-done
log(){ echo "$(date -Is) [fnc] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE; echo $$ > $W/fncompat$SFX.pid
PRED=${1:?usage: fncompat.sh BINRUN_PID}
log "waiting for binrun pid $PRED"
WAIT_MAX=86400 $W/waitproc.sh "$PRED" >> $P 2>&1
log "binrun finished; starting compatibility gate"
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml GGML_CUDA_ALLREDUCE=nccl
export GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1 LLAMA_PLE_SHARD=1
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
BPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${BPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/fncompat$SFX-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/fncompat$SFX-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/fncompat$SFX-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/fncompat$SFX-smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
: > $TSV
BASEMD5=""
for arm in $(cut -d'|' -f1 $ARMS_FILE); do
  bd=/root/build-ps-$arm
  if [ ! -x $bd/bin/llama-completion ]; then log "$arm: no build — skipped"; printf "%s\tNO-BUILD\n" "$arm" >> $TSV; continue; fi
  out=$W/fnc$SFX-$arm.txt
  LD_LIBRARY_PATH=$bd/bin:$bd/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib timeout 900 \
    $bd/bin/llama-completion -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -lm mlock --n-cpu-moe 41 \
    -fa on -ctk f16 -ctv f16 -c 4096 -n 64 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup \
    -p "Explain in two sentences why the sky is blue." > $out 2>${out%.txt}.log < /dev/null &
  BPID=$!
  wait $BPID
  rc=$?
  BPID=""
  n=$(wc -c < $out); md5=$(md5sum < $out | cut -c1-12)
  [ "$arm" = base ] && BASEMD5=$md5
  same=different; [ "$md5" = "$BASEMD5" ] && same=identical
  if [ $rc -eq 0 ] && [ "$n" -gt 60 ]; then v=PASS; else v=FAIL; fi
  printf "%s\t%s\trc=%s\tbytes=%s\t%s\t%s\n" "$arm" "$v" "$rc" "$n" "$md5" "$same" >> $TSV
  log "$arm: $v (rc=$rc, $n bytes, text $same vs base)"
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $DONE; log "=== FN COMPAT DONE ==="
