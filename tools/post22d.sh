#!/bin/bash
# After post22c: the decisive production-readiness check for the gfx906 branch — the server level (team16 profile, 1300/256 requests,
# conc 4,8,12,16) on the r2 build with the repack on and off, against production's 67.4 / 76.1 / 83.0 / 83.4. Pre-empts queue-21 and relaunches it.
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress; TAG=qwen38-27b-post22d
q() { echo "$(date -Is) $*" >> $Q; }
while [ ! -f $B/.post22c-done ]; do sleep 10; done
sleep 3; pkill -f "^/bin/bash $B/queue-21-ablation[.]sh"; pkill -f "^/bin/bash $B/ablation-bench[.]sh"; pkill -f "^/opt/llama.cpp-ablation"; pkill -f "^/opt/llama.cpp/bin/llama-"; pkill -f "^/opt/llama.cpp-mxxm-fh/bin/llama-"; pkill -f "^timeout [0-9]* /opt/llama.cpp"; sleep 5
q "=== post22d start (queue-21 pre-empted)"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/$TAG-clocks.txt
QUEUE="^/bin/bash $B/post22d[.]sh" DONEFLAG=$B/.post22d-done SMCLOG=$B/smc-power-post22d.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-post22d.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-post22d.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
OUT=$B/$TAG.md; echo "# $TAG $(date -Is): gfx906 branch (r2) at the server level, team16 profile (-np 16, 16 x 32K), 1300/256 requests; production read 67.4 / 76.1 / 83.0 / 83.4 at 4 / 8 / 12 / 16 clients" > $OUT
P=/opt/llama.cpp-gfx906-master-r2
for rp in on off; do extra=""; [ $rp = off ] && extra="--no-repack"
  LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 16 -cb -c $((16*32768)) -b 2048 -ub 2048 $extra --host 127.0.0.1 --port 8089 > $B/$TAG-$rp.log 2>&1 & S=$!
  if wait_server 8089 $S; then { echo; echo "## repack $rp"; } >> $OUT; python3 $B/server-bench.py http://127.0.0.1:8089 $TAG-$rp --conc 4,8,12,16 --gen 256 --prompt-tokens 1300 >> $OUT 2>>$B/$TAG-$rp.log; else echo "## repack $rp: server failed to start" >> $OUT; fi
  stop_server $S; q "post22d repack $rp: $(grep -E '^\| [0-9]+ ' $OUT | tail -4 | awk -F'|' '{printf "c%s %s; ", $2, $5}')"; done
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; pkill -f "smc-log[.]sh"; touch $B/.post22d-done; q "=== post22d done; relaunching queue-21"
nohup setsid /bin/bash $B/queue-21-ablation.sh > $B/queue-21-ablation.out 2>&1 &
