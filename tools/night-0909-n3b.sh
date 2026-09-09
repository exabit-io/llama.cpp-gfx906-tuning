#!/bin/bash
# n3b = n3 with DPP kept out of the flash-attention kernels: build, then A/B vs n3 (rocm0 + tp4, depth 0 and 32K, 2 rounds).
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
TAG=qwen38-27b-night0909; OUT=$B/$TAG.md; WT=/root/wt-night-n3b; N3=/opt/llama.cpp-n3-dpp; N3B=/opt/llama.cpp-n3b-dppnofa
q() { echo "$(date -Is) $*" >> $Q; }
q "=== n3b build start"; ( cd $WT && CCACHE_BASEDIR=/root bash scripts/gfx906/build.sh $N3B > $WT/build.log 2>&1 ) && q "n3b built" || { q "n3b BUILD FAILED"; exit 1; }
q "=== n3b measure start"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/n3b-clocks.txt
QUEUE="^/bin/bash $B/night-0909-n3b[.]sh" DONEFLAG=$B/.n3b-done SMCLOG=$B/smc-power-n3b.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-night0909.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-n3b.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; q "n3b KILLED"; exit 1; }; trap cleanup INT TERM
{ echo; echo "## 8. n3b = DPP everywhere except the flash-attention kernels, vs n3 ($(date -Is))"; echo "| round | build | dev | test | depth | t/s |"; echo "|---|---|---|---|---:|---:|"; } >> $OUT
for r in 1 2; do for b in $( [ $r = 1 ] && echo "n3 n3b" || echo "n3b n3" ); do P=$N3; [ $b = n3b ] && P=$N3B
  LD_LIBRARY_PATH=$P/lib timeout 1800 $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -b 2048 -ub 2048 -p 2048 -n 128 -d 0,32768 -r 2 -o md > $B/$TAG-lb8-$b-tp4-r$r.md 2>/dev/null
  LD_LIBRARY_PATH=$P/lib timeout 1800 $P/bin/llama-bench -m $M --device rocm0 -fa 1 -b 2048 -ub 2048 -p 512 -n 128 -d 0,32768 -r 2 -o md > $B/$TAG-lb8-$b-rocm0-r$r.md 2>/dev/null
  for dev in tp4 rocm0; do grep -E '^\| .*(pp|tg)[0-9]+' $B/$TAG-lb8-$b-$dev-r$r.md | while IFS='|' read -r _ rest; do
    test=$(echo "$rest" | awk -F'|' '{print $(NF-2)}' | tr -d ' '); tps=$(echo "$rest" | awk -F'|' '{print $(NF-1)}' | tr -d ' ')
    depth=0; case "$test" in *@d32768*) depth=32768;; esac; test=${test%%@*}; echo "| $r | $b | $dev | $test | $depth | $tps |" >> $OUT; done; done
  q "night0909 lb8 r$r $b: tp4 $(grep -E '^\| .*(pp|tg)[0-9]+' $B/$TAG-lb8-$b-tp4-r$r.md | awk -F'|' '{gsub(/ /,"",$(NF-2)); gsub(/ /,"",$(NF-1)); printf "%s=%s ", $(NF-2), $(NF-1)}') | rocm0 $(grep -E '^\| .*(pp|tg)[0-9]+' $B/$TAG-lb8-$b-rocm0-r$r.md | awk -F'|' '{gsub(/ /,"",$(NF-2)); gsub(/ /,"",$(NF-1)); printf "%s=%s ", $(NF-2), $(NF-1)}')"
done; done
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"
wait_ready() { local port=$1 pid=$2 i; for i in $(seq 1 900); do curl -sf http://127.0.0.1:$port/health >/dev/null && return 0; kill -0 $pid 2>/dev/null || return 1; sleep 1; done; return 1; }
{ echo; echo "| round | build | spec | w1 gen t/s | w2 gen t/s | accepted / drafted |"; echo "|---|---|---|---:|---:|---|"; } >> $OUT
for r in 1 2; do for b in $( [ $r = 1 ] && echo "n3 n3b" || echo "n3b n3" ); do P=$N3; [ $b = n3b ] && P=$N3B; LOG=$B/$TAG-mtpc-$b-r$r-srv.log
  LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M $D4 -fa on -b 2048 -ub 2048 -cb --host 127.0.0.1 --port 8089 -np 1 -c 34816 --spec-type draft-mtp --spec-draft-n-max 3 > $LOG 2>&1 & S=$!
  if wait_ready 8089 $S; then C=$B/$TAG-mtpc-$b-r$r-client.md
    python3 $B/mtp-depth-client.py http://127.0.0.1:8089 "$TAG-mtpc-$b-r$r" $LOG --context-tokens 32768 --conc 1 --gen 300 --waves 2 --warmup 300 > $C 2>>$B/$TAG-mtp.err
    w1=$(grep 'wave 1' $C | tail -1); w2=$(grep 'wave 2' $C | tail -1)
    g1=$(echo "$w1" | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}'); g2=$(echo "$w2" | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}'); acc=$(echo "$w2" | awk -F'|' '{gsub(/^ +| +$/,"",$11); print $11}')
    echo "| $r | $b | mtp3 | ${g1:--} | ${g2:--} | ${acc:--} |" >> $OUT; q "night0909 mtpc r$r $b: g2 $g2 acc $acc"
  else q "night0909 mtpc r$r $b: SERVER FAILED"; fi
  stop_server $S
done; done
echo "# n3b done $(date -Is)" >> $OUT; kill $SAMP 2>/dev/null; restore; q "night0909 n3b ALLDONE"; touch $B/.n3b-done
