#!/bin/bash
# pr-test.sh — verification for the mxxm-t/mx-llama.cpp v0.5.0 merge PR. Pass/fail and smoke only.
# Runs the merged build with the FORK'S OWN defaults (no Exabit env beyond the box's RCCL topology file,
# fine-grain PCIe for the fork's custom AllReduce, and LLAMA_PLE_SHARD=1 which Flash-Next needs to fit).
set -u
W=/root/night-20260919; R=/root/rocm-tests/bench; B=/root/build-mx-sync
M=/root/models/Qwen3.8-27B-Q8_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
FN=/root/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
P=$W/pr-test.progress; DONE=$W/.pr-test-done; O=$W/pr-test
LDP=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib
D4="--device rocm0,rocm1,rocm2,rocm3"
log(){ echo "$(date -Is) [pr] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE; mkdir -p $O; echo $$ > $W/pr-test.pid
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml HSA_FORCE_FINE_GRAIN_PCIE=1 LLAMA_PLE_SHARD=1
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
T0=""; T1=""; T2=""; T3=""; BPID=""; SPID=""
cleanup(){ trap - INT TERM EXIT; for p in "${T0:-}" "${T1:-}" "${T2:-}" "${T3:-}" "${BPID:-}" "${SPID:-}" "${SAMP:-}" "${WDOG:-}" "${SMCL:-}"; do kpid "$p"; done
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $O/clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$O/smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $O/watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $O/smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
run(){ # run NAME CMD... — foreground by PID so the trap can reach it
  local name=$1; shift
  "$@" > $O/$name.out 2> $O/$name.log < /dev/null &
  BPID=$!
  wait $BPID; local rc=$?
  BPID=""
  return $rc
}
log "1. test-backend-ops, every op, one process per device"
LD_LIBRARY_PATH=$LDP timeout 5400 $B/bin/test-backend-ops -b ROCm0 > $O/tbo-0.log 2>&1 &
T0=$!
LD_LIBRARY_PATH=$LDP timeout 5400 $B/bin/test-backend-ops -b ROCm1 > $O/tbo-1.log 2>&1 &
T1=$!
LD_LIBRARY_PATH=$LDP timeout 5400 $B/bin/test-backend-ops -b ROCm2 > $O/tbo-2.log 2>&1 &
T2=$!
LD_LIBRARY_PATH=$LDP timeout 5400 $B/bin/test-backend-ops -b ROCm3 > $O/tbo-3.log 2>&1 &
T3=$!
for i in 0 1 2 3; do
  v=T$i; wait "${!v}"; rc=$?
  log "  ROCm$i rc=$rc: $(grep -E '[0-9]+/[0-9]+ tests passed' $O/tbo-$i.log | tail -1 || echo 'NO SUMMARY')"
done
T0=""; T1=""; T2=""; T3=""
log "2. perplexity, Qwen3.8-27B Q8_0, 16K x 6 chunks, -sm tensor (reference 5.6171 +/- 0.0624)"
LD_LIBRARY_PATH=$LDP run ppl timeout 7200 $B/bin/llama-perplexity -m $M $D4 -sm tensor -ngl all -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F
log "  rc=$? $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $O/ppl.log | tail -1)"
log "3. Qwen3.8-27B greedy 64 tokens, -sm layer"
LD_LIBRARY_PATH=$LDP run layer timeout 900 $B/bin/llama-completion -m $M $D4 -sm layer -ngl all -fa on -c 4096 -n 64 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup -p "Explain in two sentences why the sky is blue."
log "  rc=$? $(wc -c < $O/layer.out) bytes: $(tr '\n' ' ' < $O/layer.out | cut -c1-200)"
log "4. Qwen3.8-27B MTP speculative decode (--spec-type draft-mtp, n_max 2), llama-server, -sm tensor"
LD_LIBRARY_PATH=$LDP $B/bin/llama-server -m $M $D4 -sm tensor -ngl all -fa on -c 8192 -np 1 --port 8091 \
  --spec-type draft-mtp --spec-draft-n-max 2 > $O/mtp-server.out 2> $O/mtp-server.log < /dev/null &
SPID=$!
if wait_server 8091 $SPID; then
  curl -s --max-time 600 http://127.0.0.1:8091/completion -H 'Content-Type: application/json' \
    -d '{"prompt":"Write a short paragraph about the history of the transistor.","n_predict":256,"temperature":0,"top_k":1,"seed":0}' > $O/mtp-response.json
  log "  response: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); t=d.get("timings",{}); print("tokens", t.get("predicted_n"), "tok/s %.2f"%t.get("predicted_per_second",0), "draft_n", t.get("draft_n"), "accepted", t.get("draft_n_accepted"), "| text:", d.get("content","")[:120].replace(chr(10)," "))' $O/mtp-response.json 2>&1)"
else
  log "  MTP server did not come up: $(tail -3 $O/mtp-server.log | tr '\n' ' ' | cut -c1-300)"
fi
kpid "$SPID"; wait "$SPID" 2>/dev/null; SPID=""; sleep 3
log "5. Qwen3.8-Flash-Next (qwen4exp) greedy 64 tokens, -sm tensor, --n-cpu-moe 41, mlock"
LD_LIBRARY_PATH=$LDP run fn timeout 900 $B/bin/llama-completion -m $FN $D4 -sm tensor -ngl all -lm mlock --n-cpu-moe 41 -fa on -c 4096 -n 64 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup -p "Explain in two sentences why the sky is blue."
log "  rc=$? $(wc -c < $O/fn.out) bytes: $(tr '\n' ' ' < $O/fn.out | cut -c1-200)"
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $DONE; log "=== PR TEST DONE ==="
