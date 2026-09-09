#!/bin/bash
# After post22d and the round-2 builds: test fork positions 33..47 (per-sample tp4 tg128 x5) to name the commit that lengthened the
# after-load warm-up; pre-empts queue-21 and relaunches it.
B=/root/rocm-tests/bench; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
q() { echo "$(date -Is) $*" >> $Q; }
while [ ! -f $B/.post22d-done ] || ! grep -q FORK-ROUND2-BUILT $B/bisect/fork-build.log; do sleep 15; done
sleep 3; pkill -f "^/bin/bash $B/queue-21-ablation[.]sh"; pkill -f "^/bin/bash $B/ablation-bench[.]sh"; pkill -f "^/opt/llama.cpp-ablation"; pkill -f "^/opt/llama.cpp/bin/llama-"; pkill -f "^/opt/llama.cpp-mxxm-fh/bin/llama-"; pkill -f "^timeout [0-9]* /opt/llama.cpp"; sleep 5
q "=== post22e start (fork bisect round 2; queue-21 pre-empted)"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/post22e-clocks.txt
QUEUE="^/bin/bash $B/post22e[.]sh" DONEFLAG=$B/.post22e-done SMCLOG=$B/smc-power-post22e.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-post22e.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-post22e.log > /dev/null 2>&1 &
{ echo; echo "## fork-history bisect round 2 (positions 33..47, oldest first)"; } >> $B/bisect/tg.md
for s in $(cat $B/bisect/fork-round2-list.txt); do P=/opt/fork-bisect/$s; [ -x $P/bin/llama-bench ] || { q "post22e missing build $s"; continue; }; q "post22e $($B/bisect/tg-test.sh $P fork-$s)"; done
kill $SAMP 2>/dev/null; restore; pkill -f "smc-log[.]sh"; touch $B/.post22e-done; q "=== post22e done; relaunching queue-21"
nohup setsid /bin/bash $B/queue-21-ablation.sh > $B/queue-21-ablation.out 2>&1 &
