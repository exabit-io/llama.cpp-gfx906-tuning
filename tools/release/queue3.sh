#!/bin/bash
# queue3.sh -- (1) the q5_1 n=1 MUL_MAT tolerance edge: 50 runs per build per die, master and candidates
#              (2) per-fold attribution of the folds-on/off gap: old master vs rebased master
set -u
W=/root/rebase-v070; R=/root/rocm-tests/bench
L(){ echo "$1/bin:$1/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib"; }
log(){ echo "$(date -Is) [q3] $*" | tee -a $W/queue3.log; }
log "queue3 start pid $$"
. $R/gpu-test-env.sh; setmax
CASE='MUL_MAT(type_a=q5_1,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0)'
rep(){ # rep NAME BUILD DEV
  local name=$1 b=$2 dev=$3 fail=0 i
  : > $W/q5_1-$name.log
  for i in $(seq 1 50); do
    LD_LIBRARY_PATH=$(L $b) timeout 120 $b/bin/test-backend-ops -b $dev -o "$CASE" >> $W/q5_1-$name.log 2>&1 || fail=$((fail+1))
  done
  log "q5_1 n=1 k=256 $name on $dev: $fail/50 failed; worst ERR $(grep -oE 'ERR = [0-9.e-]+' $W/q5_1-$name.log | sort -t= -k2 -g | tail -1)"
}
rep master /root/build-master-v070 ROCm2 &
P1=$!
rep candidates /root/build-candidates-v070 ROCm3 &
P2=$!
rep merge /root/build-merge-v070 ROCm1 &
P3=$!
wait $P1 $P2 $P3
restore
$W/fold-isolate.sh /root/build-mx-both $W/isolate-old-master; log "isolate old master rc=$?"
$W/fold-isolate.sh /root/build-master-v070 $W/isolate-master-v070; log "isolate master-v070 rc=$?"
log "=== QUEUE3 DONE ==="
