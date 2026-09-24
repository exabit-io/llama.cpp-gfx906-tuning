#!/bin/bash
# Service-axis runner: llama-server -cb + bench/chat-client.py (R2.4 session shape).
# Driven by an immutable runlist file:  tag | slots | depth | clients | server_extra | client_extra
#
# Enforces what has silently broken before:
#   T1  libs pinned to the build AND the 10.0 runtime; asserted with ldd before the first run.
#   T3  the trap handler exits.
#   D7  the client's results are VERIFIED written (a crash after the workload destroyed every
#       aggregate on 2026-09-19); a run with no output row is recorded as failed, not skipped.
#   R3.9 the server log is checked for `unused tensor blk.N.nextn.*` so an MTP run that silently
#       ran WITHOUT MTP can never be reported as an MTP result.
#   §5.1 an arrival model is mandatory: a client_extra without --arrival-rate or --ramp is refused.
set -u
RUNLIST=${RUNLIST:?need RUNLIST}
TAG=${TAG:-$(basename $RUNLIST .runlist)}
GPU_CAP=${GPU_CAP:-200}
W=/root/night-20260919; B=/root/rocm-tests/bench; Q=$B/bench-queue.progress
P=${P:-/opt/llama.cpp-gfx906-rocm10}; RT=/opt/rocm/core-10.0/lib
M=/root/models/Qwen3.8-27B-Q8_0.gguf
PORT=8091
OUT=$W/$TAG.md; LOGD=$W/$TAG-logs; DONEFLAG=$W/.$TAG-done
mkdir -p $LOGD; rm -f $DONEFLAG
log() { echo "$(date -Is) [$TAG] $*" | tee -a $W/$TAG.progress >> $Q; }
kpid() { [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100

. $B/gpu-test-env.sh
# PRODUCTION FAN VALIDATION (lead, 2026-09-19): every measurement so far pinned the fans at max
# for repeatability. In production they run under t2fanrd PWM on temperature thresholds, so the
# dies run HOTTER and fan power is lower. This variant keeps the normal curve and changes nothing
# else, so the comparison against the max-fan run is one variable.
setmax_prodfans() {
  rocm-smi --setperflevel high >/dev/null 2>&1
  cpupower frequency-set -g performance >/dev/null 2>&1
  RAPL=/sys/class/powercap/intel-rapl:0; RAPL_ORIG=$(cat $RAPL/constraint_0_power_limit_uw 2>/dev/null)
  for c in 0 1; do echo ${CPU_CAP_UW:-150000000} > $RAPL/constraint_${c}_power_limit_uw 2>/dev/null; done
  cp /root/t2fand.conf.orig /etc/t2fand.conf; systemctl restart t2fanrd
  sleep 8
}
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
export LD_LIBRARY_PATH=$P/lib:$RT
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all"
DIES="0b 0e 1b 1e"
SRV=""

cleanup() { trap - INT TERM EXIT
  kpid "${SRV:-}"; pkill -f "/bin/llama-server " 2>/dev/null
  kpid "${SAMP:-}"; kpid "${VS:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in $DIES; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT

# T1 assertion, once, before anything is measured
ldd $P/bin/llama-server | grep -qE "libggml-hip.so.0 => $P/lib" || { echo "FATAL: ggml not pinned to $P/lib"; exit 1; }
ldd $P/bin/llama-server | grep -qE "libamdhip64.so.[0-9]+ => $RT" || { echo "FATAL: HIP not pinned to $RT"; exit 1; }

setmax_prodfans; start_sampler $W/$TAG-clocks.txt
for d in $DIES; do echo $((GPU_CAP*1000000)) > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
QUEUE="^/bin/bash $W/serve[.]sh" DONEFLAG=$DONEFLAG SMCLOG=$W/$TAG-smc.log SCLK_GUARD=1 \
  nohup $B/clamp-watchdog-v2.sh >> $W/$TAG-watchdog.out 2>&1 &
nohup $B/smc-log.sh $W/$TAG-smc.log >/dev/null 2>&1 &
sleep 6
log "=== start: runlist $RUNLIST, build $(basename $P), $GPU_CAP W/die"

{ echo "# $TAG   $(date -Is)"; echo
  echo "Build \`$P\`, ROCm 10.0, tp4, \`-fa on -ctk f16 -ctv f16   `# production KV (lead 2026-09-23); q8_0 cells are superseded` -ngl all -b 2048 -ub 2048\`, $GPU_CAP W/die."
  echo "Sessions per **R2.4**: short user turn, 2K-8K generated, 30% of turns carry a tool result,"
  echo "context grows from the model's own output. Poisson arrivals (§5.1). \`bench/chat-client.py\`."
  echo "\`$(state_line)\`"; echo; } > $OUT
HDR=1

wait_ready() { local i; for i in $(seq 1 2400); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null && return 0
    kill -0 ${1:-1} 2>/dev/null || return 1; sleep 1; done; return 1; }

peak_vram() { local f=$1; : > $f
  ( declare -A pk; while true; do
      for d in $DIES; do v=$(cat /sys/bus/pci/devices/0000:$d:00.0/mem_info_vram_used 2>/dev/null || echo 0)
        [ "${v:-0}" -gt "${pk[$d]:-0}" ] && pk[$d]=$v; done
      : > $f.tmp; for d in $DIES; do echo "$d ${pk[$d]:-0}" >> $f.tmp; done; mv $f.tmp $f
      sleep 3; done ) & VS=$!; }

while IFS='|' read -r tag slots depth clients sx cx; do
  case "${tag// /}" in ''|'#'*) continue;; esac
  tag=$(echo $tag); slots=$(echo $slots); depth=$(echo $depth); clients=$(echo $clients)
  sx=$(echo $sx); cx=$(echo $cx)
  case "$cx" in *--arrival-rate*|*--ramp*) : ;;
    *) log "$tag: REFUSED — no arrival model in client args (§5.1)"; continue;; esac
  SL=$LOGD/$tag-srv.log; CL=$LOGD/$tag-client.log
  log "--- $tag: $slots slots x $((depth/1024))K, $clients clients, server_extra='${sx:-none}'"
  peak_vram $LOGD/$tag.vram
  env LD_LIBRARY_PATH=$P/lib:$RT $P/bin/llama-server -m $M $D4 -fa on -ctk f16 -ctv f16   `# production KV (lead 2026-09-23); q8_0 cells are superseded` \
      -np $slots -cb -c $(( slots * depth )) -b 2048 -ub 2048 $sx \
      --host 127.0.0.1 --port $PORT > $SL 2>&1 &
  SRV=$!
  if ! wait_ready $SRV; then
    log "$tag: SERVER FAILED — $(grep -m2 -iE 'out of memory|failed to allocate|error' $SL | tr '\n' ' ' | cut -c1-200)"
    echo "| $tag | **server failed to start** — $(grep -m1 -oiE 'out of memory|failed to allocate[^\n]{0,40}' $SL) |" >> $OUT
    kpid "$SRV"; wait $SRV 2>/dev/null; SRV=""; kpid "${VS:-}"; VS=""; continue
  fi
  log "$tag: ready — $(grep -oE 'n_slots = [0-9]+, n_ctx_slot = [0-9]+' $SL | head -1)"
  # R3.9: did MTP actually engage?
  if grep -q 'unused tensor blk.*nextn' $SL; then MTPSTATE="nextn IGNORED (MTP off)"; else MTPSTATE="nextn IN USE (MTP on)"; fi
  log "$tag: $MTPSTATE"
  case "$sx" in *spec-type*)
      case "$MTPSTATE" in *IGNORED*) log "$tag: !! MTP REQUESTED BUT NOT ACTIVE — result is NOT an MTP result";; esac;; esac
  H=""; [ $HDR = 1 ] && { H="--header"; HDR=0; }
  timeout 10800 python3 $B/chat-client.py http://127.0.0.1:$PORT $W/$TAG-$tag \
      --clients $clients $cx --seed 1 $H > $CL 2>&1
  crc=$?
  # D7: verify the client actually produced results; never report a run whose output vanished
  row=$(grep -E "^\| $TAG-$tag \|" $CL | tail -1)
  nl=$(wc -l < $W/$TAG-$tag.jsonl 2>/dev/null || echo 0)
  if [ -n "$row" ] && [ "$nl" -gt 0 ]; then
    [ -n "$H" ] && grep -E '^\| tag \||^\|---' $CL >> $OUT
    echo "$row" >> $OUT
    log "$tag: OK rc=$crc, $nl request records — $(echo "$row" | cut -c1-170)"
  else
    echo "| $tag | **client produced no result row** (rc=$crc, $nl records) — $(tail -2 $CL | tr '\n' ' ' | cut -c1-120) |" >> $OUT
    log "$tag: CLIENT FAILED rc=$crc, $nl records — $(tail -3 $CL | tr '\n' ' ' | cut -c1-200)"
  fi
  echo "  - \`$tag\`: $MTPSTATE; peak VRAM $(awk '{printf "%.1f ", $2/1073741824}' $LOGD/$tag.vram) GiB/die" >> $W/$TAG-notes.txt
  kpid "${VS:-}"; VS=""
  kpid "$SRV"; wait $SRV 2>/dev/null; SRV=""
  pkill -f "/bin/llama-server " 2>/dev/null
  sleep 15
done < $RUNLIST

{ echo; echo "Per-run notes:"; cat $W/$TAG-notes.txt 2>/dev/null
  echo; echo "Peak SMC DC total: $(sed -n 's/.*PZ0G=\([0-9]*\).*/\1/p' $W/$TAG-smc.log 2>/dev/null | sort -n | tail -1) W (envelope 1228 W)."
  echo; echo "# done $(date -Is)"; } >> $OUT
trap - INT TERM EXIT
kpid "${SAMP:-}"
for d in $DIES; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $DONEFLAG; log "=== ALLDONE"
