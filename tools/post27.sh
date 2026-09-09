#!/bin/bash
# After post26 and the r3 build: the gfx906 branch built with -funsafe-math-optimizations (the flag upstream removed on 2026-08-12 and the
# production lineage still carries): perplexity, per-sample tg, batched 8/12/16 interleaved with production, server level at 8/16 clients.
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress; P=/opt/llama.cpp-gfx906-master-r3
q() { echo "$(date -Is) $*" >> $Q; }
while [ ! -f $B/.post26-done ] || ! grep -q BUILD-R3-OK $B/build-r3.log 2>/dev/null; do sleep 30; done
q "=== post27 start (r3 = branch + unsafe-math)"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/post27-clocks.txt
QUEUE="^/bin/bash $B/post27[.]sh" DONEFLAG=$B/.post27-done SMCLOG=$B/smc-power-post27.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-post27.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-post27.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
OUT=$B/qwen38-27b-post27.md; echo "# post27 $(date -Is): branch r3 (-funsafe-math-optimizations restored), --no-repack unless noted; production for reference" > $OUT
q "post27 ppl $($B/bisect/bisect-ppl.sh $P r3-unsafe-math)"
{ echo; echo "## per-sample tg128 x5"; } >> $B/bisect/tg.md
q "post27 $($B/bisect/tg-test.sh $P r3-nr1 -nr 1)"; q "post27 $($B/bisect/tg-test.sh $P r3-nr0 -nr 0)"
{ echo; echo "## batched 8/12/16 at 2K, interleaved x2"; echo "| round | build | 8 | 12 | 16 |"; echo "|---|---|---:|---:|---:|"; } >> $OUT
for r in 1 2; do for bld in r3 prod; do if [ $bld = r3 ]; then PP=$P; X="--no-repack"; else PP=/opt/llama.cpp-prod; X=""; fi
  LD_LIBRARY_PATH=$PP/lib timeout 900 $PP/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 8,12,16 $X > $B/post27-$bld-r$r-bb.md 2>/dev/null
  cells=$(grep -E '^\|' $B/post27-$bld-r$r-bb.md | grep -v 'PP \|---' | awk -F'|' '{gsub(/ /,"",$9); printf "%s | ", $9}'); echo "| $r | $bld | $cells" >> $OUT; q "post27 bb r$r $bld: $cells"; done; done
{ echo; echo "## server level team16, conc 8,16"; } >> $OUT
for bld in r3 prod; do if [ $bld = r3 ]; then PP=$P; X="--no-repack"; else PP=/opt/llama.cpp-prod; X=""; fi
  LD_LIBRARY_PATH=$PP/lib $PP/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 16 -cb -c $((16*32768)) -b 2048 -ub 2048 $X --host 127.0.0.1 --port 8089 > $B/post27-$bld-srv.log 2>&1 & S=$!
  if wait_server 8089 $S; then { echo; echo "### $bld"; } >> $OUT; python3 $B/server-bench.py http://127.0.0.1:8089 post27-$bld --conc 8,16 --gen 256 --prompt-tokens 1300 >> $OUT 2>>$B/post27-$bld-srv.log; fi
  stop_server $S; q "post27 srv $bld: $(grep -E '^\| [0-9]+ ' $OUT | tail -2 | awk -F'|' '{printf "c%s %s; ", $2, $5}')"; done
{ echo; echo "## batched 8/12/16 with the repack on (r3)"; } >> $OUT
LD_LIBRARY_PATH=$P/lib timeout 900 $P/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 1,2,4,8,12,16,24,32 > $B/post27-r3-nr0-bb.md 2>/dev/null
cells=$(grep -E '^\|' $B/post27-r3-nr0-bb.md | grep -v 'PP \|---' | awk -F'|' '{printf "%s:%s ", $4, $9}' | tr -s ' '); echo "$cells" >> $OUT; q "post27 bb r3 repack-on: $cells"
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; pkill -f "smc-log[.]sh"; touch $B/.post27-done; q "=== post27 done"
