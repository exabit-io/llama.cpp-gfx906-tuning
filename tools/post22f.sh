#!/bin/bash
# Replaces post22e: round 2 showed the warm-up appears exactly where the fork turned the repack ON by default (position 33), and the
# tests ran with the repack on. Separate the code from the switch: positions 32/33, the pristine fork and the branch with --no-repack (-nr 1)
# and with it on, five samples each. Then relaunch queue-21.
B=/root/rocm-tests/bench; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
q() { echo "$(date -Is) $*" >> $Q; }
. $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done
pgrep -f "^/bin/bash $B/clamp-watchdog-v2[.]sh" >/dev/null || { QUEUE="^/bin/bash $B/post22f[.]sh" DONEFLAG=$B/.post22f-done SMCLOG=$B/smc-power-post22e.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-post22f.out 2>&1 & }
q "=== post22f start (repack switch vs code)"
{ echo; echo "## repack switch vs code (post22f): -nr 1 = --no-repack"; } >> $B/bisect/tg.md
q "post22f $($B/bisect/tg-test.sh /opt/fork-bisect/8253e3fbf fork-32-8253e3fbf-nr1 -nr 1)"
q "post22f $($B/bisect/tg-test.sh /opt/fork-bisect/8253e3fbf fork-32-8253e3fbf-nr0 -nr 0)"
q "post22f $($B/bisect/tg-test.sh /opt/fork-bisect/e21ccb704 fork-33-e21ccb704-nr1 -nr 1)"
q "post22f $($B/bisect/tg-test.sh /opt/fork-bisect/e21ccb704 fork-33-e21ccb704-nr0 -nr 0)"
q "post22f $($B/bisect/tg-test.sh /opt/llama.cpp-b10912 fork-b10912-nr1 -nr 1)"
q "post22f $($B/bisect/tg-test.sh /opt/llama.cpp-gfx906-master-r2 r2-nr1-again -nr 1)"
q "post22f $($B/bisect/tg-test.sh /opt/llama.cpp-gfx906-master-r2 r2-nr0-again -nr 0)"
q "post22f $($B/bisect/tg-test.sh /opt/llama.cpp-prod production-again)"
kill $SAMP 2>/dev/null; restore; pkill -f "smc-log[.]sh"; touch $B/.post22f-done; touch $B/.post22e-done; q "=== post22f done; relaunching queue-21"
nohup setsid /bin/bash $B/queue-21-ablation.sh > $B/queue-21-ablation.out 2>&1 &
