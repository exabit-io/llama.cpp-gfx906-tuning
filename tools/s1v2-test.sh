#!/bin/bash
# after b3 + the v2 build: S1 patch v2 (gfx906 config from J=32) vs v1 vs pristine master, batched 1,8,16,32 x2 interleaved + llama-bench
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress; TAG=qwen38-27b-s1v2
while [ ! -f $B/.b3-done ] || ! grep -q "BUILD-S1V2-OK\|BUILD-S1V2-FAIL" $B/build-s1-v2.log 2>/dev/null; do sleep 30; done
q() { echo "$(date -Is) $*" >> $Q; }
q "=== s1v2 start"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/s1v2-clocks.txt
QUEUE="^/bin/bash $B/s1v2-test[.]sh" DONEFLAG=$B/.s1v2-done SMCLOG=$B/smc-power-s1v2.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-s1v2.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-s1v2.log > /dev/null 2>&1 &
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md
pfx() { case $1 in up) echo /opt/llama.cpp-master;; s1) echo /opt/llama.cpp-s1-upstream;; s1v2) echo /opt/llama.cpp-s1-upstream-v2;; esac; }
{ echo "# $TAG $(date -Is): S1 patch v2 (J>=32) vs v1 (J>=8) vs pristine master 5d806aa25"; echo; echo "## batched tp4 2K -npl 1,8,16,32, two interleaved rounds"; echo "| round | build | 1 | 8 | 16 | 32 |"; echo "|---|---|---:|---:|---:|---:|"; } > $OUT
for r in 1 2; do for b in up s1 s1v2; do P=$(pfx $b); [ -x $P/bin/llama-batched-bench ] || continue
  LD_LIBRARY_PATH=$P/lib timeout 1200 $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 1,8,16,32 > $B/$TAG-$b-r$r-bb.md 2>/dev/null
  cells=$(grep -E '^\|' $B/$TAG-$b-r$r-bb.md | grep -v 'PP \|---' | awk -F'|' '{gsub(/ /,"",$9); printf "%s | ", $9}'); echo "| $r | $b | $cells" >> $OUT; q "s1v2 bb r$r $b: $cells"; done; done
{ echo; echo "## llama-bench -p 2048 -n 128 -r 3"; echo "| build | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |"; echo "|---|---:|---:|---:|---:|"; } >> $OUT
for b in s1v2 up; do P=$(pfx $b); LD_LIBRARY_PATH=$P/lib $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3,rocm0 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 128 -d 0 -r 3 -o md > $B/$TAG-$b-lb.md 2>$B/$TAG-$b-lb.err
  cells=$(grep -E '^\| qwen' $B/$TAG-$b-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$12); printf "%s | ", $12}'); echo "| $b | $cells" >> $OUT; q "s1v2 lb $b: $cells"; done
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; for c in 0 1; do echo 413000000 > $RAPL/constraint_${c}_power_limit_uw; done; pkill -f "smc-log[.]sh"; touch $B/.s1v2-done; q "=== s1v2 done"
