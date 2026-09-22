#!/bin/bash
# q4v-quality.sh — the quality gate q8_0-K/q4_0-V must pass before it can be recommended.
#
# It is now a confirmed DECODE WIN on the v0.4.1/RCCL build: +4.44% at 4x64K, +8.30% at 1x254K
# (q=0.076), prefill untouched, and 23.5% less KV. That reverses both my prediction and
# optimize.py's Q4V_SLOPE=0.092 (which says 7% slower and has the sign wrong on this build).
# A decode win from a LOSSIER cache is only real if the loss is acceptable, so: perplexity on the
# same build, same corpus, q8_0-V vs q4_0-V, 16K/6 chunks, against the 5.62 reference for v0.4.1-era
# upstream. n=2 each because ppl is near-deterministic; the pair ordering is interleaved anyway.
set -u
W=/root/night-20260919; F=/root/build-faq; R=/root/rocm-tests/bench
M=/root/models/Qwen3.8-27B-Q8_0.gguf; C=/root/models/wikitext-2-raw/wiki.test.raw
log(){ echo "$(date -Is) [q4v] $*" | tee -a $W/q4v.progress; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
[ -f "$C" ] || { echo "FATAL: corpus $C missing" >&2; exit 1; }
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml GGML_CUDA_ALLREDUCE=nccl
export GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
cleanup(){ trap - INT TERM EXIT; pkill -f "/bin/llama-perplexity " 2>/dev/null; kpid "${SAMP:-}"
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  restore; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/q4v-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
rm -f $W/.q4v-done
QUEUE_PID=$$ DONEFLAG=$W/.q4v-done SMCLOG=$W/q4v-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/q4v-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/q4v-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/q4v.tsv
ppl(){ # ppl TAG CTV REP
  local tag=$1; local ctv=$2; local rep=$3; local out=$W/q4v-$tag-$rep.log
  LD_LIBRARY_PATH=$F/bin:$F/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib timeout 5400 \
    $F/bin/llama-perplexity -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk q8_0 -ctv $ctv -f $C -c 16384 --chunks 6 -b 2048 -ub 2048 > $out 2>&1
  local v; v=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $out | tail -1 | grep -oE '[0-9.]+ \+/- [0-9.]+')
  if [ -z "${v:-}" ]; then log "PPL FAILED: $tag rep$rep"; tail -3 $out | sed 's/^/      /' | tee -a $W/q4v.progress
    printf "%s\t%s\tFAILED\n" "$tag" "$rep" >> $W/q4v.tsv; return 0; fi
  printf "%s\t%s\t%s\n" "$tag" "$rep" "$v" >> $W/q4v.tsv
  log "$tag rep$rep: PPL $v"
}
log "=== quality gate: q8_0-V vs q4_0-V, 16K/6 chunks, reference cluster ~5.62 for v0.4.1-era upstream"
for rep in 1 2; do
  if [ $rep -eq 1 ]; then ppl vq8 q8_0 $rep; ppl vq4 q4_0 $rep; else ppl vq4 q4_0 $rep; ppl vq8 q8_0 $rep; fi
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.q4v-done; log "=== Q4V QUALITY DONE ==="
