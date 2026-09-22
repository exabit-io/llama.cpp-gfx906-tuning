#!/bin/bash
# Allocation probe — records what §5.2 requires and the ladder run could not capture:
# per-die KV buffer, compute buffer and model buffer sizes, plus the layer-offload assertion.
#
# Why separate: build 11067 suppresses info-level logs, so the ladder's own logs contain only
# warnings.  Buffer sizes depend on (-c, -npl, KV type), NOT on the prompt length, so a run with
# a trivial -npp allocates byte-identical buffers at a fraction of the cost (~1-2 min/cell vs 25).
# It also settles whether `-ngl all` (which resolves to n_gpu_layers = -2) really offloads
# every layer, since a partial offload would invalidate the ladder's decode numbers.
set -u
TAG=${TAG:-probe-alloc}
CELLS=${CELLS:-"4:65536 8:196608 4:262144 8:65536 4:131072 8:98304 8:131072 4:196608"}
W=/root/night-20260919; B=/root/rocm-tests/bench; Q=$B/bench-queue.progress
P=/opt/llama.cpp-gfx906-rocm10; RT=/opt/rocm/core-10.0/lib
M=/root/models/Qwen3.8-27B-Q8_0.gguf
OUT=$W/$TAG.md; LOGD=$W/$TAG-logs; DONEFLAG=$W/.$TAG-done
mkdir -p $LOGD; rm -f $DONEFLAG
log() { echo "$(date -Is) [$TAG] $*" | tee -a $W/$TAG.progress >> $Q; }
kpid() { [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
. $B/gpu-test-env.sh
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
export LD_LIBRARY_PATH=$P/lib:$RT
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all"
DIES="0b 0e 1b 1e"
cleanup() { trap - INT TERM EXIT; pkill -f "/bin/llama-batched-bench " 2>/dev/null
  kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/$TAG-clocks.txt
QUEUE="^/bin/bash $W/probe-alloc[.]sh" DONEFLAG=$DONEFLAG SMCLOG=$W/$TAG-smc.log SCLK_GUARD=1 \
  nohup $B/clamp-watchdog-v2.sh >> $W/$TAG-watchdog.out 2>&1 &
nohup $B/smc-log.sh $W/$TAG-smc.log >/dev/null 2>&1 &
sleep 6
log "=== start; cells: $CELLS"
{ echo "# $TAG — per-die memory itemisation for the capacity ladder (R3.2)   $(date -Is)"; echo
  echo "Buffer sizes depend on \`-c\`, \`-npl\` and the KV type, not on prompt length, so these are the"
  echo "same allocations the ladder made. Budget is **31 GiB per die** (R3.2); weights ~6.3 GiB/die."; echo
  echo "| slots | depth | offload | model MiB/die | KV MiB/die | compute MiB/die | sum GiB/die | headroom GiB |"
  echo "|---:|---:|---|---:|---:|---:|---:|---:|"; } > $OUT
for c in $CELLS; do IFS=: read slots depth <<< "$c"
  ctx=$(( slots * (depth + 256) )); base=$LOGD/$slots-$depth
  log "--- probe $slots x $((depth/1024))K (ctx $ctx)"
  timeout 1800 $P/bin/llama-batched-bench -m $M $D4 -fa on -ctk q8_0 -ctv q8_0 \
     -b 2048 -ub 2048 -c $ctx -npp 64 -ntg 8 -npl $slots -v > $base.md 2>$base.log
  rc=$?
  offl=$(grep -oE 'offloaded [0-9]+/[0-9]+ layers to GPU' $base.log | head -1)
  [ -z "$offl" ] && offl=$(grep -oE 'layer\(s\) to GPU|[0-9]+/[0-9]+ layers' $base.log | head -1)
  mdl=$(grep -oE 'ROCm0 model buffer size = *[0-9.]+ MiB' $base.log | head -1 | grep -oE '[0-9.]+' | head -1)
  kvd=$(grep -oE 'ROCm0 KV buffer size = *[0-9.]+ MiB' $base.log | head -1 | grep -oE '[0-9.]+' | head -1)
  cbd=$(grep -oE 'ROCm0 compute buffer size = *[0-9.]+ MiB' $base.log | head -1 | grep -oE '[0-9.]+' | head -1)
  sum=$(awk -v a="${mdl:-0}" -v b="${kvd:-0}" -v c="${cbd:-0}" 'BEGIN{printf "%.2f",(a+b+c)/1024}')
  hdr=$(awk -v s="$sum" 'BEGIN{printf "%.2f", 31-s}')
  if [ $rc -ne 0 ] && [ -z "$kvd" ]; then
    why=$(grep -m1 -oiE 'out of memory|failed to allocate[^\n]{0,40}' $base.log | head -1)
    echo "| $slots | $((depth/1024))K | **did not allocate** (${why:-rc=$rc}) | — | — | — | — | — |" >> $OUT
    log "probe $slots x $((depth/1024))K: FAILED rc=$rc ($why)"
  else
    echo "| $slots | $((depth/1024))K | ${offl:-?} | ${mdl:-?} | ${kvd:-?} | ${cbd:-?} | $sum | $hdr |" >> $OUT
    log "probe $slots x $((depth/1024))K: ${offl:-?}; model ${mdl} KV ${kvd} compute ${cbd} MiB/die = $sum GiB"
  fi
done
{ echo; echo "# done $(date -Is)"; } >> $OUT
trap - INT TERM EXIT
kpid "${SAMP:-}"; restore; touch $DONEFLAG; log "=== ALLDONE"
