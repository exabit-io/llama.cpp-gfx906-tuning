#!/bin/bash
# After post25: perplexity of the two builds around "ggml-hip : remove -funsafe-math-optimizations" (e79e4bf66, 2026-08-12) to pin the first step.
B=/root/rocm-tests/bench; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
q() { echo "$(date -Is) $*" >> $Q; }
while [ ! -f $B/.post25-done ] || ! grep -q STEP1B-ALLBUILT $B/bisect/build-queue.log; do sleep 30; done
q "=== post26 start (step-1 pin)"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax
QUEUE="^/bin/bash $B/post26[.]sh" DONEFLAG=$B/.post26-done SMCLOG=$B/smc-power-post26.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-post26.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-post26.log > /dev/null 2>&1 &
{ echo; echo "## step-1 pin: around e79e4bf66 (remove -funsafe-math-optimizations from the HIP build)"; } >> $B/bisect/ppl.md
for s in $(sed -n 116p $B/bisect/commits-step1.txt | awk '{print $2}') e79e4bf66; do P=/opt/bisect/$s; [ -x $P/bin/llama-perplexity ] || continue; q "post26 ppl $($B/bisect/bisect-ppl.sh $P $s)"; done
restore; pkill -f "smc-log[.]sh"; touch $B/.post26-done; q "=== post26 done"
