#!/bin/bash
# night-0909-addendum.sh: after the main chain — (3b) MTP gate at 32K for n4 (adaptive MTP) paired with prod and n3, 2 rounds;
# (7) n5 = merged branch validation: test-backend-ops all ops, llama-bench tp4/rocm0 at 0/32K vs n3, perplexity 16K.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
PR=/opt/llama.cpp-prod; N3=/opt/llama.cpp-n3-dpp; N4=/opt/llama.cpp-n4-amtp; N5=/opt/llama.cpp-n5-merged
TAG=qwen38-27b-night0909; OUT=$B/$TAG.md
q() { echo "$(date -Is) $*" >> $Q; }
pfx() { case $1 in prod) echo $PR;; n3) echo $N3;; n4) echo $N4;; n5) echo $N5;; esac; }
q "=== night0909 addendum start"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/night0909-add-clocks.txt
QUEUE="^/bin/bash $B/night-0909-addendum[.]sh" DONEFLAG=$B/.night0909-add-done SMCLOG=$B/smc-power-night0909-add.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-night0909.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-night0909-add.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"
wait_ready() { local port=$1 pid=$2 i; for i in $(seq 1 900); do curl -sf http://127.0.0.1:$port/health >/dev/null && return 0; kill -0 $pid 2>/dev/null || return 1; sleep 1; done; return 1; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; pkill -f llama-server; kill ${SAMP:-} 2>/dev/null; restore; q "night0909 addendum KILLED"; exit 1; }
trap cleanup INT TERM
{ echo; echo "## 3b. MTP gate at 32K, addendum $(date -Is): n4 (adaptive MTP build) paired with prod and n3; none / draft 3 / adaptive 3..10 (n4)"
  echo "| round | build | spec | w1 gen t/s | w2 gen t/s | accepted / drafted |"; echo "|---|---|---|---:|---:|---|"; } >> $OUT
ORD=("prod n4 n3" "n3 n4 prod")
for r in 1 2; do for b in ${ORD[$((r-1))]}; do P=$(pfx $b); [ -x $P/bin/llama-server ] || continue
  specs="none mtp3"; [ $b = n4 ] && specs="none mtp3 adap10 adap6"
  for spec in $specs; do case $spec in none) SPEC="";; mtp3) SPEC="--spec-type draft-mtp --spec-draft-n-max 3";; adap10) SPEC="--spec-type draft-mtp-adaptive --spec-draft-n-max 10 --spec-draft-n-min-adaptive 3";; adap6) SPEC="--spec-type draft-mtp-adaptive --spec-draft-n-max 6 --spec-draft-n-min-adaptive 2";; esac
    LOG=$B/$TAG-mtpb-$b-$spec-r$r-srv.log
    LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M $D4 -fa on -b 2048 -ub 2048 -cb --host 127.0.0.1 --port 8089 -np 1 -c 34816 $SPEC > $LOG 2>&1 & S=$!
    if wait_ready 8089 $S; then C=$B/$TAG-mtpb-$b-$spec-r$r-client.md
      python3 $B/mtp-depth-client.py http://127.0.0.1:8089 "$TAG-mtpb-$b-$spec-r$r" $LOG --context-tokens 32768 --conc 1 --gen 300 --waves 2 --warmup 300 > $C 2>>$B/$TAG-mtp.err
      w1=$(grep 'wave 1' $C | tail -1); w2=$(grep 'wave 2' $C | tail -1)
      g1=$(echo "$w1" | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}'); g2=$(echo "$w2" | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}'); acc=$(echo "$w2" | awk -F'|' '{gsub(/^ +| +$/,"",$11); print $11}')
      echo "| $r | $b | $spec | ${g1:--} | ${g2:--} | ${acc:--} |" >> $OUT; q "night0909 mtpb r$r $b $spec: g1 $g1 g2 $g2 acc $acc"
    else echo "| $r | $b | $spec | server failed | | |" >> $OUT; q "night0909 mtpb r$r $b $spec: SERVER FAILED"; fi
    stop_server $S
  done; done; done

{ echo; echo "## 7. n5 = merged branch (upstream 2026-09-09 + mx-llama.cpp master + night-0909) validation"; } >> $OUT
if [ -x $N5/bin/llama-server ]; then
  TBO=/root/wt-night-merge/build/bin/test-backend-ops
  if [ -x $TBO ]; then LD_LIBRARY_PATH=/root/wt-night-merge/build/bin:$N5/lib timeout 3600 $TBO test -b ROCm0 > $B/$TAG-tbo-n5.log 2>&1
    echo "- n5 test-backend-ops all ops: $(grep -E '^  [0-9]+/[0-9]+ tests passed|Backend ROCm0: (OK|FAIL)' $B/$TAG-tbo-n5.log | tr '\n' ' ') $(grep -c FAIL $B/$TAG-tbo-n5.log) FAIL lines" >> $OUT; fi
  { echo; echo "| round | build | dev | test | depth | t/s |"; echo "|---|---|---|---|---:|---:|"; } >> $OUT
  for r in 1 2; do for b in $( [ $r = 1 ] && echo "n3 n5" || echo "n5 n3" ); do P=$(pfx $b)
    LD_LIBRARY_PATH=$P/lib timeout 1800 $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -b 2048 -ub 2048 -p 2048 -n 128 -d 0,32768 -r 2 -o md > $B/$TAG-lb5-$b-tp4-r$r.md 2>/dev/null
    LD_LIBRARY_PATH=$P/lib timeout 1800 $P/bin/llama-bench -m $M --device rocm0 -fa 1 -b 2048 -ub 2048 -p 512 -n 128 -d 0,32768 -r 2 -o md > $B/$TAG-lb5-$b-rocm0-r$r.md 2>/dev/null
    for dev in tp4 rocm0; do grep -E '^\| .*(pp|tg)[0-9]+' $B/$TAG-lb5-$b-$dev-r$r.md | while IFS='|' read -r _ rest; do
      test=$(echo "$rest" | awk -F'|' '{print $(NF-2)}' | tr -d ' '); tps=$(echo "$rest" | awk -F'|' '{print $(NF-1)}' | tr -d ' ')
      depth=0; case "$test" in *@d32768*) depth=32768;; esac; test=${test%%@*}; echo "| $r | $b | $dev | $test | $depth | $tps |" >> $OUT; done; done
    q "night0909 lb5 r$r $b: $(grep -E '^\| .*(pp|tg)[0-9]+' $B/$TAG-lb5-$b-tp4-r$r.md | awk -F'|' '{gsub(/ /,"",$(NF-2)); gsub(/ /,"",$(NF-1)); printf "%s=%s ", $(NF-2), $(NF-1)}')"
  done; done
  LD_LIBRARY_PATH=$N5/lib timeout 900 $N5/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f /root/models/wikitext-2-raw/wiki.test.raw > $B/$TAG-ppl-n5.log 2>&1
  echo "- n5 perplexity 16K/6: $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-ppl-n5.log | tail -1)" >> $OUT; q "night0909 ppl n5: $(grep -oE 'PPL = [0-9.]+' $B/$TAG-ppl-n5.log | tail -1)"
else echo "- n5 not built" >> $OUT; fi
echo "# addendum done $(date -Is)" >> $OUT; kill $SAMP 2>/dev/null; restore; q "night0909 addendum ALLDONE"; touch $B/.night0909-add-done
