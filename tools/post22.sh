#!/bin/bash
# After queue-22 (chain paused before queue-20): (1) the new build with --no-repack, llama-bench tp4 + rocm0; (2) perplexity of every
# bisect build in /opt/bisect; (3) relaunch queue-20 (then 21 and 23 follow on their own).
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
q() { echo "$(date -Is) $*" >> $Q; }
while [ ! -f $B/.queue-22-done ]; do [ -f $B/.clamp-detected ] && { q "post22: clamp flag; not starting"; exit 2; }; sleep 30; done
. $B/gpu-test-env.sh
for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done
q "=== post22 start"; setmax; start_sampler $B/post22-clocks.txt
QUEUE="^/bin/bash $B/post22[.]sh" DONEFLAG=$B/.post22-done SMCLOG=$B/smc-power-post22.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-post22.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-post22.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
OUT=$B/qwen38-27b-post22.md; echo "# post22 $(date -Is): the gfx906 branch build (r2, mul_mat_id fix) with and without the fork's weight repack; bisect perplexities" > $OUT
{ echo; echo "## llama-bench -p 2048 -n 128 -r 3 (tp4 + rocm0), gfx906.env"; echo "| build | repack | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |"; echo "|---|---|---:|---:|---:|---:|"; } >> $OUT
for nr in 1 0; do P=/opt/llama.cpp-gfx906-master-r2; LD_LIBRARY_PATH=$P/lib timeout 1200 $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3,rocm0 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 128 -r 3 -nr $nr -o md > $B/post22-nr$nr-lb.md 2>$B/post22-nr$nr-lb.err
  cells=$(grep -E '^\| qwen' $B/post22-nr$nr-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$(NF-1)); printf "%s | ", $(NF-1)}'); echo "| gfx906-master-r2 | $([ $nr = 1 ] && echo off || echo on) | $cells" >> $OUT; q "post22 lb nr=$nr: $cells"; done
{ echo; echo "## batched-bench decode tok/s at 2K, tp4, --no-repack (repack on: 1:25.8 2:39.8 4:127.2 8:174.4 12:141.4 16:169.2 24:219.7 32:258.6; production 1:57.9 8:174.6 12:197.1 16:202.8 32:212.7)"; } >> $OUT
P=/opt/llama.cpp-gfx906-master-r2; LD_LIBRARY_PATH=$P/lib timeout 1500 $P/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 1,2,4,8,12,16,24,32 --no-repack > $B/post22-nr1-bb.md 2>$B/post22-nr1-bb.err
grep -E '^\|' $B/post22-nr1-bb.md | grep -v 'PP \|---' | awk -F'|' '{printf "%s:%s ", $4, $9}' | tr -s ' ' >> $OUT; echo >> $OUT; q "post22 bb nr=1: $(grep -E '^\|' $B/post22-nr1-bb.md | grep -v 'PP \|---' | awk -F'|' '{printf "%s:%s ", $4, $9}' | tr -s ' ')"
{ echo; echo "## bisect perplexities (16K, 6 chunks, tp4)"; } >> $OUT
for P in /opt/bisect/*/; do P=${P%/}; [ -x $P/bin/llama-perplexity ] || continue; r=$($B/bisect/bisect-ppl.sh $P); q "post22 ppl $r"; done
cat $B/bisect/ppl.md >> $OUT
# (3) where the repacked decode stalls: kernel trace of one-die tg64 with repack on and off
{ echo; echo "## kernel trace, tp4 tg64 (llama-bench -p 0 -n 64 -r 2), repack on / off: top kernels and inter-kernel gaps"; } >> $OUT
for nr in 0 1; do T=$B/trace-repack-nr$nr; rm -rf $T; P=/opt/llama.cpp-gfx906-master-r2
  LD_LIBRARY_PATH=$P/lib timeout 900 rocprofv3 --kernel-trace --stats -f csv -d $T -o rp -- $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 0 -n 64 -r 2 -nr $nr -o md > $B/post22-trace-nr$nr.md 2>$B/post22-trace-nr$nr.err
  { echo "### repack $([ $nr = 0 ] && echo on || echo off): $(grep -E '^\| qwen' $B/post22-trace-nr$nr.md | awk -F'|' '{print $(NF-1)}')"; python3 $B/m1-analyze.py $T rp --first 8 --last-skip 2 2>&1 | head -40; echo; head -8 $T/rp_kernel_stats.csv | cut -c1-150; } >> $OUT 2>&1; q "post22 trace nr=$nr done"; done
kill $SAMP 2>/dev/null; restore; pkill -f "smc-log[.]sh"; touch $B/.post22-done; q "=== post22 done; relaunching queue-20"
nohup setsid /bin/bash $B/queue-20-b10912.sh > $B/queue-20-b10912.out 2>&1 &
