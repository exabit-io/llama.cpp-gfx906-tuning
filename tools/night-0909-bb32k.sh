#!/bin/bash
# stage 2b: batched decode at 32K depth without -pps (the shared prompt trips the M-RoPE position check on Qwen3.8): each sequence
# reads its own 32K prompt. -npl 8,16 for r2, n2, n3 (n1 = r2 kernels), one round rotated twice = 2 rounds.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
TAG=qwen38-27b-night0909; OUT=$B/$TAG.md
q() { echo "$(date -Is) $*" >> $Q; }
pfx() { case $1 in r2) echo /opt/llama.cpp-gfx906-master-r2;; n2) echo /opt/llama.cpp-n2-tile;; n3) echo /opt/llama.cpp-n3-dpp;; prod) echo /opt/llama.cpp-prod;; esac; }
q "=== night0909 bb32k start"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/bb32k-clocks.txt
QUEUE="^/bin/bash $B/night-0909-bb32k[.]sh" DONEFLAG=$B/.bb32k-done SMCLOG=$B/smc-power-bb32k.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-night0909.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-bb32k.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; q "bb32k KILLED"; exit 1; }; trap cleanup INT TERM
{ echo; echo "## 2b. llama-batched-bench -npp 32768 -ntg 128 -npl 8,16 (own prompt per sequence; -pps fails the M-RoPE position check), tp4, $(date -Is)"
  echo "| round | build | slots | pp t/s | tg t/s | total |"; echo "|---|---|---:|---:|---:|---:|"; } >> $OUT
ORD=("r2 n2 n3 prod" "prod n3 n2 r2")
for r in 1 2; do for b in ${ORD[$((r-1))]}; do P=$(pfx $b); [ -x $P/bin/llama-batched-bench ] || continue
  LD_LIBRARY_PATH=$P/lib timeout 3600 $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 526336 -npp 32768 -ntg 128 -npl 8,16 > $B/$TAG-bb32k-$b-r$r.md 2>$B/$TAG-bb32k-$b-r$r.err
  grep -E '^\| *32768 ' $B/$TAG-bb32k-$b-r$r.md | awk -F'|' -v r=$r -v b=$b '{gsub(/ /,""); print "| " r " | " b " | " $4 " | " $7 " | " $9 " | " $11 " |"}' >> $OUT
  q "night0909 bb32k r$r $b: $(grep -E '^\| *32768 ' $B/$TAG-bb32k-$b-r$r.md | awk -F'|' '{gsub(/ /,""); printf "%s:%s ", $4, $9}')"
done; done
echo "# bb32k done $(date -Is)" >> $OUT; kill $SAMP 2>/dev/null; restore; q "night0909 bb32k ALLDONE"; touch $B/.bb32k-done
