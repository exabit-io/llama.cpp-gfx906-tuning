#!/bin/bash
# After post22: the tp4 single-stream instability of the gfx906 branch persists with --no-repack (37.9 +/- 16.4), on the pristine fork
# too, not on upstream, not on one die. Knob sweep on the r2 build (tp4, -nr 1, gfx906.env base): which fork feature stalls it.
# Skips queue-20 (b10912-validate: same fork state, lower value now) and releases queue-21/23 by touching its done flag.
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress; P=/opt/llama.cpp-gfx906-master-r2
q() { echo "$(date -Is) $*" >> $Q; }
while [ ! -f $B/.post22-done ]; do sleep 15; done
sleep 5; pkill -f "^/bin/bash $B/queue-20-b10912[.]sh"; pkill -f "^/bin/bash $B/b10912-validate[.]sh"; pkill -f "^/opt/llama.cpp-b10912[^ ]*/bin/"; pkill -f "^/opt/llama.cpp-prod/bin/"; sleep 5
q "=== post22b start (queue-20 skipped)"
. $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/post22b-clocks.txt
QUEUE="^/bin/bash $B/post22b[.]sh" DONEFLAG=$B/.post22b-done SMCLOG=$B/smc-power-post22b.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-post22b.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-post22b.log > /dev/null 2>&1 &
OUT=$B/qwen38-27b-post22b.md; echo "# post22b $(date -Is): gfx906 branch (r2) tp4 single-stream instability, knob sweep; llama-bench -p 0 -n 128 -r 3 --no-repack, gfx906.env base" > $OUT
echo "| variant | tp4 tg128 |" >> $OUT; echo "|---|---:|" >> $OUT
run() { local name=$1; shift; ( set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib "$@"
  timeout 600 $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 0 -n 128 -r 3 -nr 1 -o md > $B/post22b-$name.md 2>$B/post22b-$name.err )
  local c; c=$(grep -E '^\| qwen' $B/post22b-$name.md | awk -F'|' '{gsub(/^ +| +$/,"",$(NF-1)); print $(NF-1)}' | tail -1); echo "| $name | ${c:-fail} |" >> $OUT; q "post22b $name: ${c:-fail}"; }
run base X=1
run no-graphs GGML_CUDA_DISABLE_GRAPHS=1
run token-graph-off GGML_META_TOKEN_GRAPH=0
run parallel-dispatch-off GGML_META_PARALLEL_DISPATCH=0
run graph-reuse-off LLAMA_GRAPH_REUSE_DISABLE=1
run graph-opt-0 GGML_CUDA_GRAPH_OPT=0
run hwqueues-8 GPU_MAX_HW_QUEUES=8
run no-dedicated-cpy GGML_META_NO_DEDICATED_CPY=1
run no-custom-ar GGML_ENABLE_CUSTOM_AR=0
run all-fork-off GGML_META_TOKEN_GRAPH=0 GGML_META_PARALLEL_DISPATCH=0 GGML_ENABLE_CUSTOM_AR=0 LLAMA_GRAPH_REUSE_DISABLE=1
run base-repeat X=1
{ echo; echo "## bisect anchors: upstream at the fork base 0f3a71be1 and the fork pristine b10912 (ppl 16K/6)"; } >> $OUT
for P in /opt/bisect/0f3a71be1 /opt/llama.cpp-b10912; do [ -x $P/bin/llama-perplexity ] || continue; r=$($B/bisect/bisect-ppl.sh $P $(basename $P)); echo "| $r |" >> $OUT; q "post22b ppl $r"; done
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; pkill -f "smc-log[.]sh"; touch $B/.post22b-done; touch $B/.queue-20-done; q "=== post22b done; queue-20 done flag set for queue-21/23"
