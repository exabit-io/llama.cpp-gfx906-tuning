#!/bin/bash
# After post22b: per-sample tp4 tg128 (x5) on the branch (repack on/off), production, the pristine fork and pristine upstream; then round 1
# of the fork-history bisect (7 midpoint builds in /opt/fork-bisect) once built; then relaunch queue-21 (ablation) -> 23.
B=/root/rocm-tests/bench; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
q() { echo "$(date -Is) $*" >> $Q; }
while [ ! -f $B/.post22b-done ]; do sleep 15; done
q "=== post22c start"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/post22c-clocks.txt
QUEUE="^/bin/bash $B/post22c[.]sh" DONEFLAG=$B/.post22c-done SMCLOG=$B/smc-power-post22c.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-post22c.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-post22c.log > /dev/null 2>&1 &
q "post22c ppl $($B/bisect/bisect-ppl.sh /opt/llama.cpp-b10837 b10837-5202104)"
{ echo; echo "## per-sample tp4 tg128 x5 (post22c)"; } >> $B/bisect/tg.md
q "post22c $($B/bisect/tg-test.sh /opt/llama.cpp-gfx906-master-r2 r2-repack-on -nr 0)"
q "post22c $($B/bisect/tg-test.sh /opt/llama.cpp-gfx906-master-r2 r2-repack-off -nr 1)"
q "post22c $($B/bisect/tg-test.sh /opt/llama.cpp-prod production)"
q "post22c $($B/bisect/tg-test.sh /opt/llama.cpp-b10912 fork-b10912-pristine)"
q "post22c $($B/bisect/tg-test.sh /opt/llama.cpp-master upstream-master-pristine)"
q "post22c $($B/bisect/tg-test.sh /opt/llama.cpp-mxxm-fh production-0907)"
while ! grep -q FORK-ALLBUILT $B/bisect/fork-build.log 2>/dev/null; do sleep 30; done
{ echo; echo "## fork-history bisect round 1 (first-parent b10254..b10912, positions 16/32/48/64/80/96/112)"; } >> $B/bisect/tg.md
for P in /opt/fork-bisect/*/; do P=${P%/}; [ -x $P/bin/llama-bench ] || continue; q "post22c $($B/bisect/tg-test.sh $P fork-$(basename $P))"; done
kill $SAMP 2>/dev/null; restore; pkill -f "smc-log[.]sh"; touch $B/.post22c-done; q "=== post22c done; relaunching queue-21"
nohup setsid /bin/bash $B/queue-21-ablation.sh > $B/queue-21-ablation.out 2>&1 &
