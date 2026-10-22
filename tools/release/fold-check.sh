#!/bin/bash
# fold-check.sh BUILD OUTDIR -- numerical check of the decode-sized fusions (norm+q8, add+norm, GDN producer fold) in the
# exabit-io master terms: perplexity at ubatch 32 (the folds only engage at <= 64 rows) with every fold on, then all off.
# The two must agree to rounding; a broken fold moves PPL by whole units.
set -u
B=$1; O=$2
R=/root/rocm-tests/bench
M=/root/models/Qwen3.8-27B-Q8_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
mkdir -p $O; P=$O/progress; DONE=$O/.done; rm -f $DONE; echo $$ > $O/pid
LDP=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib
D4="--device rocm0,rocm1,rocm2,rocm3"
log(){ echo "$(date -Is) [fold] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml HSA_FORCE_FINE_GRAIN_PCIE=1
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
BPID=""
cleanup(){ trap - INT TERM EXIT; for p in "${BPID:-}" "${SAMP:-}" "${WDOG:-}" "${SMCL:-}"; do kpid "$p"; done
  restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $O/clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$O/smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $O/watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $O/smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
UB=${UB:-16}
ppl(){ # ppl NAME ENV... -- 2K x 8 chunks at ubatch UB (16: every fold engages, the norm+q8 one needs <= 16 rows)
  local name=$1; shift
  env "$@" LD_LIBRARY_PATH=$LDP timeout 3600 $B/bin/llama-perplexity -v -m $M $D4 -sm tensor -ngl all -fa on -b $UB -ub $UB -c 2048 --chunks 8 -f $F > $O/$name.out 2> $O/$name.log < /dev/null &
  BPID=$!; wait $BPID; local rc=$?; BPID=""
  log "  $name rc=$rc $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $O/$name.log | tail -1); fusion lines: $(grep -E 'fusion active|prefuse' $O/$name.log | sed 's/.*: //' | tr '\n' ';' | cut -c1-200)"
}
log "fold check on $B (ubatch $UB); $(state_line)"
ppl folds-on   GGML_CUDA_FOLD_CHECK=1
ppl folds-on2  GGML_CUDA_FOLD_CHECK=2
ppl folds-off  GGML_CUDA_NORM_Q8=0 GGML_CUDA_ADD_NORM=0 GGML_CUDA_GDN_PREFUSE=0
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore; touch $DONE; log "=== FOLD CHECK DONE ==="
