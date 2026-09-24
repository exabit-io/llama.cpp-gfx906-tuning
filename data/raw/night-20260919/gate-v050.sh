#!/bin/bash
# gate-v050.sh — round 0 correctness gate for the v0.5.0 substrate (RE-RE-SURVEY plan §4). Pass/fail only.
#   1. test-backend-ops, every op, one process per die (a -b filter skips the other dies, so run all four)
#   2. perplexity 16K/6 on the 27B vs the reference 5.6171 +/- 0.0624 (cluster 5.5969-5.6448): a
#      correctness comparison at the reference's own settings, never a performance result
#   3. Flash-Next (qwen4exp) at production flags, greedy 64 tokens: loads and generates coherent text
set -u
W=/root/night-20260919; R=/root/rocm-tests/bench; B=/root/build-v050m-substrate
M=/root/models/Qwen3.8-27B-Q8_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
FN=/root/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
P=$W/gate-v050.progress; DONE=$W/.gate-v050-done
LDP=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib
log(){ echo "$(date -Is) [g050] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE; echo $$ > $W/gate-v050.pid
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml GGML_CUDA_ALLREDUCE=nccl HSA_FORCE_FINE_GRAIN_PCIE=1 LLAMA_PLE_SHARD=1
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
T0=""; T1=""; T2=""; T3=""; BPID=""
cleanup(){ trap - INT TERM EXIT; for p in "${T0:-}" "${T1:-}" "${T2:-}" "${T3:-}" "${BPID:-}" "${SAMP:-}" "${WDOG:-}" "${SMCL:-}"; do kpid "$p"; done
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/gate-v050-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/gate-v050-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/gate-v050-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/gate-v050-smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
log "1. test-backend-ops on ROCm0..3"
LD_LIBRARY_PATH=$LDP timeout 5400 $B/bin/test-backend-ops -b ROCm0 > $W/gate-v050-tbo-0.log 2>&1 &
T0=$!
LD_LIBRARY_PATH=$LDP timeout 5400 $B/bin/test-backend-ops -b ROCm1 > $W/gate-v050-tbo-1.log 2>&1 &
T1=$!
LD_LIBRARY_PATH=$LDP timeout 5400 $B/bin/test-backend-ops -b ROCm2 > $W/gate-v050-tbo-2.log 2>&1 &
T2=$!
LD_LIBRARY_PATH=$LDP timeout 5400 $B/bin/test-backend-ops -b ROCm3 > $W/gate-v050-tbo-3.log 2>&1 &
T3=$!
fail=0
for i in 0 1 2 3; do
  v=T$i; wait "${!v}"; rc=$?
  line=$(grep -E '[0-9]+/[0-9]+ tests passed' $W/gate-v050-tbo-$i.log | tail -1)
  nf=$(grep -c 'FAIL' $W/gate-v050-tbo-$i.log)
  log "  ROCm$i rc=$rc: ${line:-NO SUMMARY} (FAIL lines: $nf)"
  [ $rc -ne 0 ] && fail=1
done
T0=""; T1=""; T2=""; T3=""
log "test-backend-ops: $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
log "2. perplexity 16K/6 (reference 5.6171 +/- 0.0624)"
LD_LIBRARY_PATH=$LDP timeout 7200 $B/bin/llama-perplexity -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
  -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $W/gate-v050-ppl.log 2>&1 &
BPID=$!
wait $BPID; rc=$?; BPID=""
log "  ppl rc=$rc -> $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $W/gate-v050-ppl.log | tail -1)"
log "3. Flash-Next compatibility (production flags, greedy 64 tokens)"
LD_LIBRARY_PATH=$LDP timeout 900 $B/bin/llama-completion -m $FN --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
  -lm mlock --n-cpu-moe 41 -fa on -ctk f16 -ctv f16 -c 4096 -n 64 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup \
  -p "Explain in two sentences why the sky is blue." > $W/gate-v050-fn.txt 2>$W/gate-v050-fn.log < /dev/null &
BPID=$!
wait $BPID; rc=$?; BPID=""
log "  flash-next rc=$rc, $(wc -c < $W/gate-v050-fn.txt) bytes: $(tr '\n' ' ' < $W/gate-v050-fn.txt | cut -c1-240)"
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $DONE; log "=== GATE v0.5.0 DONE ==="
