#!/bin/bash
# tile-thr-v110.sh -- static: the head-256 tile instance of master-v110 (llama.cpp v1.1.0, the kernel after upstream's
# #32901) recompiled by ROCm 10.1's clang 24 with max-ilp at every scheduler VGPR threshold, and scratch reloads inside
# loops counted per kernel (codegen/tilescan.py). Reference row: the same source by ROCm 10.0's clang 23 (no rule).
# Rule written before the first compile, the one of 2026-11-16 (report 2026-11-14-rocm-10.1.md s.5.10): a threshold is
# usable if NO kernel of the object has a reload in a loop that the clang 23 object does not have there; among those,
# the candidates go to the timing cells (design points, MTP at depth). No GPU, no timing.
set -u
C=/root/rocm101-migration-20261114/codegen; B1=/root/build-master-v110-101; B0=/root/build-master-v110
SUF=fattn-tile-instance-f16-f16-dkq256-dv256.cu
cd /root/rebase-v110/tile/thr || exit 1
PIDS=""
KPAT=flash_attn_tile python3 $C/trycc.py $B1 $SUF tnone --drop 'amdgpu-vgpr-threshold-percent' > try-tnone.out 2>&1 &
PIDS="$PIDS $!"
for t in ${THRS:-95 92 90 88 85 82 80 78 75 70 60 50}; do
  KPAT=flash_attn_tile python3 $C/trycc.py $B1 $SUF t$t --drop 'amdgpu-vgpr-threshold-percent' -mllvm -amdgpu-vgpr-threshold-percent=$t > try-t$t.out 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p; done
O0=$(find $B0 -name "$SUF.o" | head -1)
{ echo "rows in order: clang 23 (ROCm 10.0, no rule), clang 24 no rule, then clang 24 at threshold ${THRS:-95 92 90 88 85 82 80 78 75 70 60 50}"
  TMPDIR=/root/rebase-v110/tile python3 $C/tilescan.py $O0 try-tnone.o $(for t in ${THRS:-95 92 90 88 85 82 80 78 75 70 60 50}; do echo try-t$t.o; done); } > ../tilescan-thresholds-v110.txt 2>&1
echo "done: $(ls try-t*.o | wc -l) objects"
