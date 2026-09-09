#!/bin/bash
# after s1b-test: does the 8-cell loss after the 32-cell persist across later 8-cells, and does 24 trigger it? r2-nr1, a3, prod
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress; TAG=qwen38-27b-b3
while [ ! -f $B/.s1b-test-done ]; do sleep 30; done
q() { echo "$(date -Is) $*" >> $Q; }
q "=== b3 start"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/b3-clocks.txt
QUEUE="^/bin/bash $B/b3-transition[.]sh" DONEFLAG=$B/.b3-done SMCLOG=$B/smc-power-b3.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-b3.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-b3.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md
{ echo "# $TAG $(date -Is): batched-bench tp4 2K -npl 32,8,8,8,16,8,24,8,32,8,8,8 (-c 69632): persistence of the after-32 loss, and 24 as a trigger"; echo "| build | 32 | 8 | 8 | 8 | 16 | 8 | 24 | 8 | 32 | 8 | 8 | 8 |"; echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"; } > $OUT
for b in r2-nr1 a3 prod; do case $b in r2-nr1) P=/opt/llama.cpp-gfx906-master-r2; X="--no-repack";; a3) P=/opt/llama.cpp-gfx906-s1b-a3; X="";; prod) P=/opt/llama.cpp-prod; X="";; esac
  LD_LIBRARY_PATH=$P/lib timeout 1500 $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 32,8,8,8,16,8,24,8,32,8,8,8 $X > $B/$TAG-$b-bb.md 2>/dev/null
  cells=$(grep -E '^\|' $B/$TAG-$b-bb.md | grep -v 'PP \|---' | awk -F'|' '{gsub(/ /,"",$9); printf "%s | ", $9}'); echo "| $b | $cells" >> $OUT; q "b3 $b: $cells"; done
{ echo; echo "## same on a 35K cache (-c 34816 = 16 x 2176): -npl 16,8,8,8,12,8,16,8,8"; echo "| build | 16 | 8 | 8 | 8 | 12 | 8 | 16 | 8 | 8 |"; echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"; } >> $OUT
for b in r2-nr1 prod; do case $b in r2-nr1) P=/opt/llama.cpp-gfx906-master-r2; X="--no-repack";; prod) P=/opt/llama.cpp-prod; X="";; esac
  LD_LIBRARY_PATH=$P/lib timeout 1500 $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 34816 -npp 2048 -ntg 128 -npl 16,8,8,8,12,8,16,8,8 $X > $B/$TAG-c35-$b-bb.md 2>/dev/null
  cells=$(grep -E '^\|' $B/$TAG-c35-$b-bb.md | grep -v 'PP \|---' | awk -F'|' '{gsub(/ /,"",$9); printf "%s | ", $9}'); echo "| $b | $cells" >> $OUT; q "b3 c35 $b: $cells"; done
# server level repeat: a3 vs production interleaved x2 (the first pass read production 64.9 at 8 clients, a 15% outlier against its history)
{ echo; echo "## server level team16 conc 8,16 repeat, a3 / prod / a3 / prod"; } >> $OUT
for b in a3 prod a3b prodb; do case $b in a3*) P=/opt/llama.cpp-gfx906-s1b-a3;; prod*) P=/opt/llama.cpp-prod;; esac
  LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M $D4 -fa on -np 16 -cb -c $((16*32768)) -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 > $B/$TAG-E-$b-srv.log 2>&1 & S=$!
  if wait_server 8089 $S; then { echo; echo "### $b"; } >> $OUT; python3 $B/server-bench.py http://127.0.0.1:8089 $TAG-E-$b --conc 8,16 --gen 256 --prompt-tokens 1300 >> $OUT 2>>$B/$TAG-E-$b-srv.log; fi
  stop_server $S; q "b3 srv $b: $(grep -E '^\| [0-9]+ ' $OUT | tail -2 | awk -F'|' '{printf "c%s %s; ", $2, $5}')"; done
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; for c in 0 1; do echo 413000000 > $RAPL/constraint_${c}_power_limit_uw; done; pkill -f "smc-log[.]sh"; touch $B/.b3-done; q "=== b3 done"
