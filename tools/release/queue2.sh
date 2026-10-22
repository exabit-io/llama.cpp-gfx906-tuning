#!/bin/bash
# queue2.sh -- verification of the rebased exabit-io branches (master-v070, candidates-v070) and the dio GPU-path trace
set -u
W=/root/rebase-v070; R=/root/rocm-tests/bench
M=/root/models/Qwen3.8-27B-Q8_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
BM=/root/build-master-v070; BC=/root/build-candidates-v070; BG=/root/build-merge-v070
L(){ echo "$1/bin:$1/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib"; }
log(){ echo "$(date -Is) [q2] $*" | tee -a $W/queue2.log; }
log "queue2 start pid $$"
DONE=$W/.queue2-done; rm -f $DONE
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/queue2-smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $W/queue2-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/queue2-smc.log >/dev/null 2>&1 &
SMCL=$!
cleanup(){ trap - INT TERM EXIT; kill $WDOG $SMCL 2>/dev/null; . $R/gpu-test-env.sh; restore; touch $DONE; log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM
# 1. -lm dio on the GPU path of the merge build: is the file opened O_DIRECT, and does any read fall back?
mkdir -p $W/dio
LD_LIBRARY_PATH=$(L $BG) timeout 600 strace -f -e trace=openat -o $W/dio/strace-gpu-merge.txt $BG/bin/llama-completion -m $M \
  --device rocm0,rocm1,rocm2,rocm3 -sm layer -ngl 99 -fa on -lm dio -c 2048 -n 8 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup \
  -p "The capital of France is" > $W/dio/gpu-merge.out 2> $W/dio/gpu-merge.log < /dev/null
log "dio gpu path rc=$? O_DIRECT opens=$(grep -c 'Q8_0.gguf", O_RDONLY|O_DIRECT' $W/dio/strace-gpu-merge.txt) fallbacks=$(grep -c 'Falling back' $W/dio/gpu-merge.log) out='$(tr '\n' ' ' < $W/dio/gpu-merge.out | cut -c1-60)'"
# 2. fold check at ubatch 16: the pre-rebase master (reference), the reconciled master, the rebased master
$W/fold-check.sh /root/build-mx-both $W/fold-old-master;  log "fold old master rc=$?"
$W/fold-check.sh /root/build-reconcile $W/fold-reconcile16; log "fold reconcile rc=$?"
$W/fold-check.sh $BM $W/fold-master-v070;                 log "fold master-v070 rc=$?"
# 3. test-backend-ops: master on ROCm0/1, candidates on ROCm2/3
. $R/gpu-test-env.sh; setmax
LD_LIBRARY_PATH=$(L $BM) timeout 5400 $BM/bin/test-backend-ops -b ROCm0 > $W/tbo-master-0.log 2>&1 & P0=$!
LD_LIBRARY_PATH=$(L $BM) timeout 5400 $BM/bin/test-backend-ops -b ROCm1 > $W/tbo-master-1.log 2>&1 & P1=$!
LD_LIBRARY_PATH=$(L $BC) timeout 5400 $BC/bin/test-backend-ops -b ROCm2 > $W/tbo-cand-2.log 2>&1 & P2=$!
LD_LIBRARY_PATH=$(L $BC) timeout 5400 $BC/bin/test-backend-ops -b ROCm3 > $W/tbo-cand-3.log 2>&1 & P3=$!
for p in $P0 $P1 $P2 $P3; do wait $p; done
for f in tbo-master-0 tbo-master-1 tbo-cand-2 tbo-cand-3; do
  log "$f: $(grep -E '[0-9]+/[0-9]+ tests passed' $W/$f.log | tail -1) failed: $(grep -E 'FAIL' $W/$f.log | grep -v 'tests passed' | head -3 | tr '\n' ' ' | cut -c1-300)"
done
restore
# 4. the PR suite minus test-backend-ops on the rebased master (RCCL on), and perplexity on candidates
SKIP_TBO=1 SRC=/root/wt-master-v070 $W/pr-test.sh $BM $W/pr-test-master-v070; log "pr-test master-v070 rc=$?"
. $R/gpu-test-env.sh; setmax
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml HSA_FORCE_FINE_GRAIN_PCIE=1
LD_LIBRARY_PATH=$(L $BC) timeout 7200 $BC/bin/llama-perplexity -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $W/ppl-candidates.out 2> $W/ppl-candidates.log < /dev/null
log "candidates ppl rc=$? $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $W/ppl-candidates.log | tail -1)"
restore
trap - INT TERM
kill $WDOG $SMCL 2>/dev/null; touch $DONE
log "=== QUEUE2 DONE ==="
