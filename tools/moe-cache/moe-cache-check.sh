#!/bin/bash
# moe-cache-check.sh [OUT] -- does upstream's MoE expert cache (--moe-cache-mib, --moe-cache-cpu-share; in our tree
# since v1.0.0) run on this box with the experts in host RAM, and does it engage. Installed package 1.1.0.
# Functional check: no rate printed here is a result (4K context; REQUIREMENTS R2.2).
#
# Why (lead, 2026-11-28): bigger MoE models (GLM, DeepSeek, Kimi) with the experts in DRAM and the hot ones in
# VRAM, and whether 1.5 TB of RAM per machine pays. Two models on disk are larger than what the dies hold next to
# their context: DeepSeek-V4.1-Flash UD-IQ1_S and Qwen3.8-Flash-Next UD-Q4_K_XL.
#
# Known from the source before any run (src/llama-moe-cache.cpp, docs/moe-cache.md):
#   - the cache THROWS under -sm tensor ("MoE cache does not support tensor parallelism"): layer split only;
#   - it serves ubatches of at most 32 tokens (decode, drafts); large batches upload the experts they use;
#   - only layers that keep ALL their experts in host memory are cached; the budget is split over the dies.
#
# Runs per model, all -sm layer over four dies, -cmoe (every expert in host RAM), greedy, 96 tokens:
#   cmoe    no cache: the CPU computes the experts (what -cmoe meant before the cache)
#   cache   --moe-cache-mib 32768, share 0: hits on the die, misses uploaded
#   share   --moe-cache-mib 32768 --moe-cache-cpu-share 0.5: half of the misses on the CPU
#
# READING, WRITTEN BEFORE THE FIRST RUN.
#   (M1) Each run loads and generates: rc, the placement lines and the cache's own lines (size, slots, share of the
#        experts, "tokens worth") are reported. A run that fails is reported with its last log lines.
#   (M2) The cache engages if its statistics line shows hits for ubatch <= 8; hit rate and uploaded MiB are reported.
#   (M3) Text: the three texts of a model are printed. A cache text that is not a sensible continuation of the prompt
#        is a defect (the fork repacks weights on upload; a cached expert in the wrong layout would show here).
#        Equality with the no-cache text is reported and NOT required: without the cache the CPU computes the experts,
#        with it the die does, and the two round differently.
#   (M4) The eval rates of the three runs are printed as a first hint only. If cache or share is not faster than cmoe
#        here, that is said; the real measurement is at 64K with moe-cache-tune.py and is a separate script.
set -u
W=/root/moe-cache; R=/root/rocm-tests/bench; O=${1:-$W/check}; mkdir -p $O
PKG=/opt/mx-llama.cpp
DSM=/root/models/DeepSeek-V4.1-Flash-1017-UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf
FNM=/root/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
D4="--device rocm0,rocm1,rocm2,rocm3"; MIB=${MIB:-32768}
P=$O/progress; DONE=$O/.done; rm -f $DONE
log(){ echo "$(date -Is) [moe] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
for f in $DSM $FNM; do [ -f $f ] || { log "FATAL: $f missing"; exit 1; }; done
for p in llama-server llama-batched-b llama-perplexit llama-completi test-backend-op llama-cli cmake ninja; do
  pgrep -x $p >/dev/null && { log "FATAL: $p is running"; exit 1; }
done
log "package $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906); cache $MIB MiB; RAM free $(awk '/MemAvailable/{printf "%.0f GiB", $2/1048576}' /proc/meminfo); load $(cut -d' ' -f1-3 /proc/loadavg)"
. $R/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1 LLAMA_PLE_SHARD=1
BPID=""
# the run is timeout -> llama-completion in the background of this shell: signal the child of BPID too
cleanup(){ trap - INT TERM EXIT; [ -n "$BPID" ] && { pkill -TERM -P "$BPID" 2>/dev/null; kpid "$BPID"; }; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $O/clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$O/smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $O/watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $O/smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
log "$(state_line)"
one(){ # one NAME MODEL PROMPT extra args...
  local name=$1 m=$2 prompt=$3; shift 3
  local t0; t0=$(date +%s)
  env -u LD_LIBRARY_PATH timeout 3600 $PKG/bin/llama-completion -m $m $D4 -sm layer -fa on -c 4096 -n 96 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup "$@" -p "$prompt" > $O/$name.out 2> $O/$name.log < /dev/null &
  BPID=$!; wait $BPID; local rc=$?; BPID=""
  log "$name rc=$rc ($(( $(date +%s) - t0 )) s), $(wc -c < $O/$name.out) bytes, md5 $(md5sum < $O/$name.out | cut -c1-8): $(tr '\n' ' ' < $O/$name.out | cut -c1-330)"
  log "$name rates (functional): $(grep -E 'prompt eval time|       eval time' $O/$name.log | tail -2 | sed 's/.*= *//' | tr '\n' ' ' | cut -c1-220)"
  local c; c=$(grep -hE 'MoE cache|llama_moe_cache|slots \(' $O/$name.log | sed 's/^[0-9.]* [A-Z] //' | tr '\n' ';' | cut -c1-900)
  [ -n "$c" ] && log "$name cache: $c"
  [ $rc = 0 ] || log "$name FAILED: $(grep -v 'New LWP\|^\[Thread\|^warning:' $O/$name.log | tail -4 | tr '\n' ' ' | cut -c1-500)"
  return $rc
}
DSP="The transistor was invented in 1947 at Bell Labs. In the decades that followed,"
FNP="Explain in two sentences why the sky is blue."
for model in ${MODELS:-ds fn}; do
  case $model in ds) m=$DSM; pr=$DSP; ex="";; fn) m=$FNM; pr=$FNP; ex="-ngl all -lm mlock";; esac
  one $model-cmoe  $m "$pr" $ex -cmoe
  one $model-cache $m "$pr" $ex -cmoe --moe-cache-mib $MIB
  one $model-share $m "$pr" $ex -cmoe --moe-cache-mib $MIB --moe-cache-cpu-share 0.5
  a=$(md5sum < $O/$model-cmoe.out | cut -c1-8); b=$(md5sum < $O/$model-cache.out | cut -c1-8); c=$(md5sum < $O/$model-share.out | cut -c1-8)
  log "$model (M3) texts: cmoe $a, cache $b ($([ $a = $b ] && echo identical || echo differs)), share $c ($([ $a = $c ] && echo identical || echo differs))"
done
trap - INT TERM EXIT
kpid $SAMP; kpid $SMCL; touch $DONE; sleep 6; kpid $WDOG; restore
log "=== MOE-CACHE-CHECK DONE ==="
