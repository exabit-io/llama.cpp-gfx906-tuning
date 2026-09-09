#!/bin/bash
# after s1v2 + the batched-bench builds: intermittent slow 8-row cells across the fork's history (positions 16..112), production and the branch
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress; TAG=qwen38-27b-bisect8
while [ ! -f $B/.s1v2-done ] || [ ! -f $B/bisect/.bb-builds-done ]; do sleep 30; done
q() { echo "$(date -Is) $*" >> $Q; }
q "=== bisect8 start"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/bisect8-clocks.txt
QUEUE="^/bin/bash $B/bisect-8row[.]sh" DONEFLAG=$B/.bisect8-done SMCLOG=$B/smc-power-bisect8.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-bisect8.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-bisect8.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md
{ echo "# $TAG $(date -Is): batched-bench tp4 2K, twelve 8-row cells in one process (-npl 8 x12, -c 69632); production repeats 174.0 +/- 0.3, the branch shows intermittent 155-170 cells"; echo "| build (fork position) | cells | min | max | n < 171 |"; echo "|---|---|---:|---:|---:|"; } > $OUT
run() { local name=$1 P=$2; shift 2; [ -x $P/bin/llama-batched-bench ] || { echo "| $name | no binary | | | |" >> $OUT; return; }
  LD_LIBRARY_PATH=$P/lib timeout 1500 $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 8,8,8,8,8,8,8,8,8,8,8,8 "$@" > $B/$TAG-$name-bb.md 2>/dev/null
  local cells; cells=$(grep -E '^\|' $B/$TAG-$name-bb.md | grep -v 'PP \|---' | awk -F'|' '{gsub(/ /,"",$9); printf "%s ", $9}')
  echo "| $name | $cells | $(echo $cells | tr ' ' '\n' | sort -n | head -1) | $(echo $cells | tr ' ' '\n' | sort -n | tail -1) | $(echo $cells | tr ' ' '\n' | awk '$1<171' | wc -l) |" >> $OUT; q "bisect8 $name: $cells"; }
run prod /opt/llama.cpp-prod
run branch-r2-nr1 /opt/llama.cpp-gfx906-master-r2 --no-repack
run fork-b10912-nr1 /opt/llama.cpp-b10912 --no-repack
for e in "16:803f00bb7" "32:8253e3fbf" "33:e21ccb704" "34:97e14020c" "48:115a32c1e" "64:a65e71eb2" "80:def64b335" "96:9435cfcc4" "112:16628b027"; do pos=${e%%:*}; sha=${e##*:}; X=""; [ $pos -ge 33 ] && X="--no-repack"; run "fork-$pos-$sha" /opt/fork-bisect/$sha $X; done
run prod-repeat /opt/llama.cpp-prod
run branch-r2-nr1-repeat /opt/llama.cpp-gfx906-master-r2 --no-repack
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; for c in 0 1; do echo 413000000 > $RAPL/constraint_${c}_power_limit_uw; done; pkill -f "smc-log[.]sh"; touch $B/.bisect8-done; q "=== bisect8 done"
