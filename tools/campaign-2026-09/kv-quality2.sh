#!/bin/bash
# kv-quality2.sh — quality gate for the KV types the 2026-09-22/23 sweep put in contention.
#
# The sweep found the V-width curve is NON-MONOTONIC (q5_0/q5_1 slower than q8_0; q4_1 faster than
# q4_0), that f16 is fastest everywhere by 18-39%, and that q4_0-q4_0 -- quantising K as well -- is
# second fastest AND smallest. Speed rankings mean nothing until quality is checked, and K
# quantisation is where the risk lives: this gate exists mainly to find out whether q4_0-K is
# acceptable at all.
#
# It is now a confirmed DECODE WIN on the v0.4.1/RCCL build: +4.44% at 4x64K, +8.30% at 1x254K
# (q=0.076), prefill untouched, and 23.5% less KV. That reverses both my prediction and
# optimize.py's Q4V_SLOPE=0.092 (which says 7% slower and has the sign wrong on this build).
# A decode win from a LOSSIER cache is only real if the loss is acceptable, so: perplexity on the
# same build, same corpus, q8_0-V vs q4_0-V, 16K/6 chunks, against the 5.62 reference for v0.4.1-era
# upstream. n=2 each because ppl is near-deterministic; the pair ordering is interleaved anyway.
set -u
W=/root/night-20260919; F=/root/build-faq-allquants; R=/root/rocm-tests/bench
M=/root/models/Qwen3.8-27B-Q8_0.gguf; C=/root/models/wikitext-2-raw/wiki.test.raw
log(){ echo "$(date -Is) [kvq] $*" | tee -a $W/kvq.progress; }
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
setmax; start_sampler $W/kvq-clocks.txt
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
rm -f $W/.kvq-done
QUEUE_PID=$$ DONEFLAG=$W/.kvq-done SMCLOG=$W/kvq-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/kvq-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/kvq-smc.log >/dev/null 2>&1 &
sleep 6
: > $W/kvq.tsv
ppl(){ # ppl TAG CTK CTV REP
  local tag=$1; local ctk=$2; local ctv=$3; local rep=$4; local out=$W/kvq-$tag-$rep.log
  LD_LIBRARY_PATH=$F/bin:$F/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib timeout 5400 \
    $F/bin/llama-perplexity -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk $ctk -ctv $ctv -f $C -c 16384 --chunks 6 -b 2048 -ub 2048 > $out 2>&1
  local v; v=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $out | tail -1 | grep -oE '[0-9.]+ \+/- [0-9.]+')
  if [ -z "${v:-}" ]; then log "PPL FAILED: $tag rep$rep"; tail -3 $out | sed 's/^/      /' | tee -a $W/kvq.progress
    printf "%s\t%s\tFAILED\n" "$tag" "$rep" >> $W/kvq.tsv; return 0; fi
  printf "%s\t%s\t%s\n" "$tag" "$rep" "$v" >> $W/kvq.tsv
  log "$tag rep$rep: PPL $v"
}
log "=== quality gate, 16K/6 chunks, reference cluster ~5.62 for v0.4.1-era upstream"
# ppl TAG CTK CTV REP
for rep in 1 2; do
  ppl f16     f16  f16  $rep
  ppl q8q8    q8_0 q8_0 $rep
  ppl q8q41   q8_0 q4_1 $rep
  ppl q8q4    q8_0 q4_0 $rep
  ppl q4q4    q4_0 q4_0 $rep
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null; pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
restore; touch $W/.kvq-done; log "=== KV QUALITY DONE ==="
