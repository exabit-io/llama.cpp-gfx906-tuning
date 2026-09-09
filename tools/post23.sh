#!/bin/bash
# After queue-23: perplexity of the step-1 bisect builds (upstream 2026-08-05 .. 09-02, seven midpoints) to place the first 0.27% move.
B=/root/rocm-tests/bench; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
q() { echo "$(date -Is) $*" >> $Q; }
while [ ! -f $B/.queue-23-done ] || ! grep -q STEP1-ALLBUILT $B/bisect/build-queue.log; do sleep 30; done
q "=== post23 start (step-1 perplexity bisect)"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/post23-clocks.txt
QUEUE="^/bin/bash $B/post23[.]sh" DONEFLAG=$B/.post23-done SMCLOG=$B/smc-power-post23.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-post23.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-post23.log > /dev/null 2>&1 &
# (0) promotion repeat: branch --no-repack vs production, interleaved twice: batched 8/12/16 at 2K and the team16 server level
M=/root/models/Qwen3.8-27B-Q8_0.gguf; set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
OUT=$B/qwen38-27b-post23-promo.md; echo "# post23 promotion repeat $(date -Is): branch r2 --no-repack vs production, interleaved x2" > $OUT
{ echo; echo "## batched-bench decode tok/s at 2K, tp4 -npl 8,12,16"; echo "| round | build | 8 | 12 | 16 |"; echo "|---|---|---:|---:|---:|"; } >> $OUT
for r in 1 2; do for bld in r2 prod; do if [ $bld = r2 ]; then P=/opt/llama.cpp-gfx906-master-r2; X="--no-repack"; else P=/opt/llama.cpp-prod; X=""; fi
  LD_LIBRARY_PATH=$P/lib timeout 900 $P/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 8,12,16 $X > $B/post23-promo-$bld-r$r-bb.md 2>/dev/null
  cells=$(grep -E '^\|' $B/post23-promo-$bld-r$r-bb.md | grep -v 'PP \|---' | awk -F'|' '{gsub(/ /,"",$9); printf "%s | ", $9}'); echo "| $r | $bld | $cells" >> $OUT; q "post23 promo bb r$r $bld: $cells"; done; done
{ echo; echo "## server level, team16 (-np 16, 16 x 32K), 1300/256, conc 8,16"; } >> $OUT
for r in 1 2; do for bld in r2 prod; do if [ $bld = r2 ]; then P=/opt/llama.cpp-gfx906-master-r2; X="--no-repack"; else P=/opt/llama.cpp-prod; X=""; fi
  LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 16 -cb -c $((16*32768)) -b 2048 -ub 2048 $X --host 127.0.0.1 --port 8089 > $B/post23-promo-$bld-r$r-srv.log 2>&1 & S=$!
  if wait_server 8089 $S; then { echo; echo "### round $r $bld"; } >> $OUT; python3 $B/server-bench.py http://127.0.0.1:8089 post23-promo-$bld-r$r --conc 8,16 --gen 256 --prompt-tokens 1300 >> $OUT 2>>$B/post23-promo-$bld-r$r-srv.log; fi
  stop_server $S; q "post23 promo srv r$r $bld: $(grep -E '^\| [0-9]+ ' $OUT | tail -2 | awk -F'|' '{printf "c%s %s; ", $2, $5}')"; done; done
{ echo; echo "## step-1 bisect (upstream 360e1349f 2026-08-05 = stock b10288, 5.5969 .. 0f3a71be1 2026-09-02, 5.6118)"; } >> $B/bisect/ppl.md
for s in $(cat $B/bisect/step1-list.txt); do P=/opt/bisect/$s; [ -x $P/bin/llama-perplexity ] || continue; q "post23 ppl $($B/bisect/bisect-ppl.sh $P $s)"; done
kill $SAMP 2>/dev/null; restore; pkill -f "smc-log[.]sh"; touch $B/.post23-done; q "=== post23 done"
