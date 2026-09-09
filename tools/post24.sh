#!/bin/bash
# After post23 and the ablation rebuild: rerun the tile-table ablation (TODO 12) with all five builds and the UD-Q6_K / UD-Q4_K_M models.
B=/root/rocm-tests/bench; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
q() { echo "$(date -Is) $*" >> $Q; }
while [ ! -f $B/.post23-done ] || ! grep -q ABLATION-REBUILT $B/ablation-rebuild.log 2>/dev/null; do sleep 30; done
q "=== post24 start (ablation rerun)"; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done
QUEUE="^/bin/bash $B/post24[.]sh" DONEFLAG=$B/.post24-done SMCLOG=$B/smc-power-post24.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-post24.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-post24.log > /dev/null 2>&1 &
mv $B/qwen38-27b-ablation.md $B/qwen38-27b-ablation-first-try.md 2>/dev/null
$B/ablation-bench.sh > $B/ablation-bench-2.out 2>&1; q "post24 ablation rc=$?"
pkill -f "smc-log[.]sh"; touch $B/.post24-done; q "=== post24 done"
