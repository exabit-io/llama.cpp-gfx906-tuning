#!/bin/bash
# night-0909-ab.sh — the fork-survey A/Bs at 32K depth (guide BENCHMARKS-TODO 16/17, reports/2026-09-09-fork-survey.md s.6).
# Builds: r2 = /opt/llama.cpp-gfx906-master-r2 (kernels of gfx906 HEAD, the baseline), n1 = + out_ids fix (kernel-identical to r2:
# the noise floor), n2 = + GCN head-256 tile row, n3 = + DPP reductions, n4 = + adaptive MTP (PR 27210), prod = /opt/llama.cpp-prod.
# Branch builds run with the repack ON (default) and gfx906.env (custom AR + gate); prod with gfx906.env.
# Stages: 0 test-backend-ops (n2 FA, n3 all)  1 llama-bench tp4 + rocm0 at depth 0/32768 (2 rounds, rotated)  2 batched-bench -pps
# 32K depth 1/8/12/16 slots (2 rounds)  3 MTP gate at 32K: prod/r2/n4 x none/mtp3(/adaptive), 2 rounds  4 out_ids check: -np 2 two
# concurrent 32K clients on r2 vs n1  5 server level 32K prompts, 16 slots, conc 8/16: prod/n3 (2 rounds)  6 perplexity 16K n2/n3.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
PR=/opt/llama.cpp-prod; R2=/opt/llama.cpp-gfx906-master-r2; N1=/opt/llama.cpp-n1-outids; N2=/opt/llama.cpp-n2-tile; N3=/opt/llama.cpp-n3-dpp; N4=/opt/llama.cpp-n4-amtp
TAG=qwen38-27b-night0909; OUT=$B/$TAG.md
q() { echo "$(date -Is) $*" >> $Q; }
pfx() { case $1 in prod) echo $PR;; r2) echo $R2;; n1) echo $N1;; n2) echo $N2;; n3) echo $N3;; n4) echo $N4;; esac; }
q "=== night0909 start"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/night0909-clocks.txt
QUEUE="^/bin/bash $B/night-0909-ab[.]sh" DONEFLAG=$B/.night0909-done SMCLOG=$B/smc-power-night0909.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-night0909.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-night0909.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"
wait_ready() { local port=$1 pid=$2 i; for i in $(seq 1 900); do curl -sf http://127.0.0.1:$port/health >/dev/null && return 0; kill -0 $pid 2>/dev/null || return 1; sleep 1; done; return 1; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; pkill -f llama-server; kill ${SAMP:-} 2>/dev/null; restore; q "night0909 KILLED"; exit 1; }
trap cleanup INT TERM
echo "# $TAG  $(date -Is)  fork-survey A/Bs at 32K depth. $(state_line)" > $OUT
for b in r2 n1 n2 n3 n4 prod; do P=$(pfx $b); echo "- $b = $P: llama-server $(sha256sum $P/bin/llama-server 2>/dev/null | cut -c1-16), libggml-hip $(sha256sum $P/lib/libggml-hip.so* 2>/dev/null | head -1 | cut -c1-16)" >> $OUT; done

# ---- 0. test-backend-ops
{ echo; echo "## 0. test-backend-ops (ROCm0)"; } >> $OUT
for spec in "n2 /root/wt-night-n2-tile FLASH_ATTN_EXT" "n3 /root/wt-night-n3-dpp ALL"; do set -- $spec; b=$1; WT=$2; op=$3
  TBO=$WT/build/bin/test-backend-ops; [ -x $TBO ] || { ( cd $WT && cmake --build build --target test-backend-ops -j56 >/dev/null 2>&1 ); }
  if [ -x $TBO ]; then O=""; [ $op != ALL ] && O="-o $op"
    LD_LIBRARY_PATH=$WT/build/bin:$(pfx $b)/lib timeout 3600 $TBO test -b ROCm0 $O > $B/$TAG-tbo-$b.log 2>&1
    res=$(grep -E '^  [0-9]+/[0-9]+ tests passed|Backend ROCm0: (OK|FAIL)' $B/$TAG-tbo-$b.log | tr '\n' ' ')
    echo "- $b ($op): ${res:-no summary, exit $?} $(grep -c 'FAIL' $B/$TAG-tbo-$b.log) FAIL lines" >> $OUT; q "night0909 tbo $b: $res"
  else echo "- $b: test-backend-ops not built" >> $OUT; fi
done

# ---- 1. llama-bench at depth 0 and 32768, tp4 and rocm0
{ echo; echo "## 1. llama-bench -p 2048 -n 128 -d 0,32768 (tp4, -r 2) and -p 512 -n 128 -d 0,32768 (rocm0, -r 2); gfx906.env; repack on"
  echo "| round | build | dev | test | depth | t/s |"; echo "|---|---|---|---|---:|---:|"; } >> $OUT
ORD1=("r2 n1 n2 n3" "n3 n2 n1 r2")
for r in 1 2; do for b in ${ORD1[$((r-1))]}; do P=$(pfx $b); [ -x $P/bin/llama-bench ] || continue
  LD_LIBRARY_PATH=$P/lib timeout 1800 $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -b 2048 -ub 2048 -p 2048 -n 128 -d 0,32768 -r 2 -o md > $B/$TAG-lb-$b-tp4-r$r.md 2>$B/$TAG-lb-$b-tp4-r$r.err
  LD_LIBRARY_PATH=$P/lib timeout 1800 $P/bin/llama-bench -m $M --device rocm0 -fa 1 -b 2048 -ub 2048 -p 512 -n 128 -d 0,32768 -r 2 -o md > $B/$TAG-lb-$b-rocm0-r$r.md 2>$B/$TAG-lb-$b-rocm0-r$r.err
  for dev in tp4 rocm0; do grep -E '^\| .*(pp|tg)[0-9]+' $B/$TAG-lb-$b-$dev-r$r.md | while IFS='|' read -r _ rest; do
      test=$(echo "$rest" | awk -F'|' '{print $(NF-2)}' | tr -d ' '); tps=$(echo "$rest" | awk -F'|' '{print $(NF-1)}' | tr -d ' ')
      depth=0; case "$test" in *@d32768*) depth=32768;; esac; test=${test%%@*}
      echo "| $r | $b | $dev | $test | $depth | $tps |" >> $OUT; done; done
  q "night0909 lb r$r $b: $(grep -E '^\| .*(pp|tg)[0-9]+' $B/$TAG-lb-$b-tp4-r$r.md | awk -F'|' '{gsub(/ /,"",$(NF-2)); gsub(/ /,"",$(NF-1)); printf "%s=%s ", $(NF-2), $(NF-1)}')"
done; done

# ---- 2. batched-bench, shared 32K prompt, decode at depth
{ echo; echo "## 2. llama-batched-bench -pps -npp 32768 -ntg 128 -npl 1,8,12,16 (tp4): decode at 32K depth"
  echo "| round | build | slots | pp t/s (32K, once) | tg t/s | total |"; echo "|---|---|---:|---:|---:|---:|"; } >> $OUT
for r in 1 2; do for b in ${ORD1[$((r-1))]}; do P=$(pfx $b); [ -x $P/bin/llama-batched-bench ] || continue
  LD_LIBRARY_PATH=$P/lib timeout 1800 $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 36864 -pps -npp 32768 -ntg 128 -npl 1,8,12,16 > $B/$TAG-bb-$b-r$r.md 2>/dev/null
  grep -E '^\| *32768 ' $B/$TAG-bb-$b-r$r.md | awk -F'|' -v r=$r -v b=$b '{gsub(/ /,""); print "| " r " | " b " | " $4 " | " $7 " | " $9 " | " $11 " |"}' >> $OUT
  q "night0909 bb r$r $b: $(grep -E '^\| *32768 ' $B/$TAG-bb-$b-r$r.md | awk -F'|' '{gsub(/ /,""); printf "%s:%s ", $4, $9}')"
done; done

# ---- 3. MTP gate at 32K (as s1b-test A0, warmed): prod / r2 / n4, drafting off, draft 3, and adaptive 3..10 on n4
{ echo; echo "## 3. MTP gate at 32K: llama-server -np 1, prompt 32768, 300 generated, two waves (wave 2 = decode only); adaptive = --spec-type draft-mtp-adaptive --spec-draft-n-max 10 --spec-draft-n-min-adaptive 3"
  echo "| round | build | spec | w1 gen t/s | w2 gen t/s | accepted / drafted |"; echo "|---|---|---|---:|---:|---|"; } >> $OUT
ORD3=("prod r2 n4" "n4 prod r2")
for r in 1 2; do for b in ${ORD3[$((r-1))]}; do P=$(pfx $b); [ -x $P/bin/llama-server ] || continue
  specs="none mtp3"; [ $b = n4 ] && specs="none mtp3 adap10"
  for spec in $specs; do case $spec in none) SPEC="";; mtp3) SPEC="--spec-type draft-mtp --spec-draft-n-max 3";; adap10) SPEC="--spec-type draft-mtp-adaptive --spec-draft-n-max 10 --spec-draft-n-min-adaptive 3";; esac
    LOG=$B/$TAG-mtp-$b-$spec-r$r-srv.log
    LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M $D4 -fa on -b 2048 -ub 2048 -cb --host 127.0.0.1 --port 8089 -np 1 -c 34816 $SPEC > $LOG 2>&1 & S=$!
    if wait_ready 8089 $S; then C=$B/$TAG-mtp-$b-$spec-r$r-client.md
      python3 $B/mtp-depth-client.py http://127.0.0.1:8089 "$TAG-mtp-$b-$spec-r$r" $LOG --context-tokens 32768 --conc 1 --gen 300 --waves 2 --warmup 300 > $C 2>>$B/$TAG-mtp.err
      w1=$(grep 'wave 1' $C | tail -1); w2=$(grep 'wave 2' $C | tail -1)
      g1=$(echo "$w1" | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}'); g2=$(echo "$w2" | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}'); acc=$(echo "$w2" | awk -F'|' '{gsub(/^ +| +$/,"",$11); print $11}')
      echo "| $r | $b | $spec | ${g1:--} | ${g2:--} | ${acc:--} |" >> $OUT; q "night0909 mtp r$r $b $spec: g1 $g1 g2 $g2 acc $acc"
    else echo "| $r | $b | $spec | server failed | | |" >> $OUT; q "night0909 mtp r$r $b $spec: SERVER FAILED"; fi
    stop_server $S
  done; done; done

# ---- 4. out_ids check: two concurrent 32K clients on -np 2 with draft 3, r2 (bug) vs n1 (fix); acceptance per request from the server log
{ echo; echo "## 4. out_ids: -np 2, two concurrent 32K clients, draft-mtp n-max 3; acceptance lines per request (the bug: one slot at 0.03-0.10)"
  echo "| build | w1 gen t/s (agg) | w2 gen t/s (agg) | acceptance lines (wave 2) |"; echo "|---|---:|---:|---|"; } >> $OUT
for b in r2 n1; do P=$(pfx $b); [ -x $P/bin/llama-server ] || continue; LOG=$B/$TAG-outids-$b-srv.log
  LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M $D4 -fa on -b 2048 -ub 2048 -cb --host 127.0.0.1 --port 8089 -np 2 -c 69632 --spec-type draft-mtp --spec-draft-n-max 3 > $LOG 2>&1 & S=$!
  if wait_ready 8089 $S; then C=$B/$TAG-outids-$b-client.md
    python3 $B/mtp-depth-client.py http://127.0.0.1:8089 "$TAG-outids-$b" $LOG --context-tokens 32768 --conc 2 --gen 300 --waves 2 --warmup 300 > $C 2>>$B/$TAG-mtp.err
    w1=$(grep 'wave 1' $C | tail -1 | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}'); w2=$(grep 'wave 2' $C | tail -1 | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}')
    accs=$(grep -oE 'acceptance = [0-9.]+ \( *[0-9]+ accepted / *[0-9]+' $LOG | tail -4 | sed 's/acceptance = //; s/ *accepted//' | tr '\n' ';')
    echo "| $b | ${w1:--} | ${w2:--} | $accs |" >> $OUT; q "night0909 outids $b: $w1 / $w2 acc $accs"
  else echo "| $b | server failed | | |" >> $OUT; fi
  stop_server $S
done

# ---- 5. server level at 32K prompts: prod vs n3, 16 slots, conc 8 and 16
{ echo; echo "## 5. server level: llama-server -np 16 -c 16x33K, server-bench.py --prompt-tokens 32768 --gen 256 --conc 8,16; two rounds rotated"
  echo "| round | build | conc | reqs | agg gen t/s | total tok/s | per-req gen t/s | TTFT s |"; echo "|---|---|---:|---:|---:|---:|---:|---:|"; } >> $OUT
ORD5=("prod n3" "n3 prod")
for r in 1 2; do for b in ${ORD5[$((r-1))]}; do P=$(pfx $b); [ -x $P/bin/llama-server ] || continue; LOG=$B/$TAG-srv32k-$b-r$r-srv.log
  LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M $D4 -fa on -np 16 -cb -c $((16*33792)) -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 > $LOG 2>&1 & S=$!
  if wait_ready 8089 $S; then C=$B/$TAG-srv32k-$b-r$r.md
    timeout 7200 python3 $B/server-bench.py http://127.0.0.1:8089 $TAG-srv32k-$b-r$r --conc 8,16 --gen 256 --prompt-tokens 32768 --min-req 16 > $C 2>>$B/$TAG-srv32k.err
    grep -E '^\| *(8|16) ' $C | awk -F'|' -v r=$r -v b=$b '{gsub(/^ +| +$/,"",$2); gsub(/^ +| +$/,"",$3); gsub(/^ +| +$/,"",$5); gsub(/^ +| +$/,"",$6); gsub(/^ +| +$/,"",$8); gsub(/^ +| +$/,"",$9); print "| " r " | " b " | " $2 " | " $3 " | " $5 " | " $6 " | " $8 " | " $9 " |"}' >> $OUT
    q "night0909 srv32k r$r $b: $(grep -E '^\| *(8|16) ' $C | awk -F'|' '{gsub(/ /,"",$2); gsub(/ /,"",$5); printf "c%s=%s ", $2, $5}')"
  else echo "| $r | $b | server failed | | | | | |" >> $OUT; q "night0909 srv32k r$r $b: SERVER FAILED"; fi
  stop_server $S
done; done

# ---- 6. perplexity 16K / 6 chunks (reference: r2 5.6173, prod 5.5969)
{ echo; echo "## 6. perplexity 16K/6 chunks tp4 (r2 = 5.6173)"; echo "| build | ppl |"; echo "|---|---|"; } >> $OUT
for b in n2 n3; do P=$(pfx $b); [ -x $P/bin/llama-perplexity ] || continue
  LD_LIBRARY_PATH=$P/lib timeout 900 $P/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f /root/models/wikitext-2-raw/wiki.test.raw > $B/$TAG-ppl-$b.log 2>&1
  fe=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-ppl-$b.log | tail -1); echo "| $b | ${fe:-FAIL} |" >> $OUT; q "night0909 ppl $b: ${fe:-FAIL}"
done

echo "# done $(date -Is)" >> $OUT; kill $SAMP 2>/dev/null; restore; q "night0909 ALLDONE"; touch $B/.night0909-done
