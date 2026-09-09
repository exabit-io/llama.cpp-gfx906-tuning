#!/bin/bash
# final candidate = night-merge (upstream 2026-09-09 + mx-llama.cpp + out_ids fix + DPP + measured tile row + adaptive MTP + S1b a3):
# build /opt/llama.cpp-gfx906-20260909, then validate vs production at 32K: test-backend-ops, llama-bench (tp4 + rocm0, 0/32K),
# batched 8/16 at 32K, MTP gate (none / draft 3), perplexity. Result appended to qwen38-27b-night0909.md (section 9).
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
WT=/root/wt-night-merge; F=/opt/llama.cpp-gfx906-20260909; PR=/opt/llama.cpp-prod; TAG=qwen38-27b-night0909; OUT=$B/$TAG.md
q() { echo "$(date -Is) $*" >> $Q; }
q "=== final build start $(git -C $WT rev-parse --short HEAD)"
( cd $WT && CCACHE_BASEDIR=/root bash scripts/gfx906/build.sh $F > $WT/build-final.log 2>&1 && cmake --build build --target test-backend-ops -j56 >> $WT/build-final.log 2>&1 ) && q "final built" || { q "final BUILD FAILED"; tail -20 $WT/build-final.log >> $Q; exit 1; }
q "=== final measure start"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/final-clocks.txt
QUEUE="^/bin/bash $B/night-0909-final[.]sh" DONEFLAG=$B/.final-done SMCLOG=$B/smc-power-final.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-night0909.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-final.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"
wait_ready() { local port=$1 pid=$2 i; for i in $(seq 1 900); do curl -sf http://127.0.0.1:$port/health >/dev/null && return 0; kill -0 $pid 2>/dev/null || return 1; sleep 1; done; return 1; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; pkill -f llama-server; kill ${SAMP:-} 2>/dev/null; restore; q "final KILLED"; exit 1; }; trap cleanup INT TERM
pfx() { case $1 in prod) echo $PR;; final) echo $F;; esac; }
{ echo; echo "## 9. final candidate $(git -C $WT rev-parse --short HEAD) = /opt/llama.cpp-gfx906-20260909 vs production, $(date -Is)"; } >> $OUT
LD_LIBRARY_PATH=$WT/build/bin:$F/lib timeout 3600 $WT/build/bin/test-backend-ops test -b ROCm0 > $B/$TAG-tbo-final.log 2>&1
echo "- test-backend-ops all ops: $(grep -E '^  [0-9]+/[0-9]+ tests passed|Backend ROCm0: (OK|FAIL)' $B/$TAG-tbo-final.log | tr '\n' ' ') $(grep -c FAIL $B/$TAG-tbo-final.log) FAIL lines" >> $OUT; q "final tbo: $(grep -E 'tests passed' $B/$TAG-tbo-final.log | tail -1)"
{ echo; echo "| round | build | dev | test | depth | t/s |"; echo "|---|---|---|---|---:|---:|"; } >> $OUT
for r in 1 2; do for b in $( [ $r = 1 ] && echo "prod final" || echo "final prod" ); do P=$(pfx $b)
  LD_LIBRARY_PATH=$P/lib timeout 1800 $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -b 2048 -ub 2048 -p 2048 -n 128 -d 0,32768 -r 2 -o md > $B/$TAG-lb9-$b-tp4-r$r.md 2>/dev/null
  LD_LIBRARY_PATH=$P/lib timeout 1800 $P/bin/llama-bench -m $M --device rocm0 -fa 1 -b 2048 -ub 2048 -p 512 -n 128 -d 0,32768 -r 2 -o md > $B/$TAG-lb9-$b-rocm0-r$r.md 2>/dev/null
  for dev in tp4 rocm0; do grep -E '^\| .*(pp|tg)[0-9]+' $B/$TAG-lb9-$b-$dev-r$r.md | while IFS='|' read -r _ rest; do
    test=$(echo "$rest" | awk -F'|' '{print $(NF-2)}' | tr -d ' '); tps=$(echo "$rest" | awk -F'|' '{print $(NF-1)}' | tr -d ' ')
    depth=0; case "$test" in *@d32768*) depth=32768;; esac; test=${test%%@*}; echo "| $r | $b | $dev | $test | $depth | $tps |" >> $OUT; done; done
  q "final lb r$r $b: tp4 $(grep -E '^\| .*(pp|tg)[0-9]+' $B/$TAG-lb9-$b-tp4-r$r.md | awk -F'|' '{gsub(/ /,"",$(NF-2)); gsub(/ /,"",$(NF-1)); printf "%s=%s ", $(NF-2), $(NF-1)}') | rocm0 $(grep -E '^\| .*(pp|tg)[0-9]+' $B/$TAG-lb9-$b-rocm0-r$r.md | awk -F'|' '{gsub(/ /,"",$(NF-2)); gsub(/ /,"",$(NF-1)); printf "%s=%s ", $(NF-2), $(NF-1)}')"
done; done
{ echo; echo "| build | slots | pp t/s | tg t/s (32K depth, own prompts) |"; echo "|---|---:|---:|---:|"; } >> $OUT
for b in final prod; do P=$(pfx $b)
  LD_LIBRARY_PATH=$P/lib timeout 3600 $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 526336 -npp 32768 -ntg 128 -npl 8,16 > $B/$TAG-bb9-$b.md 2>/dev/null
  grep -E '^\| *32768 ' $B/$TAG-bb9-$b.md | awk -F'|' -v b=$b '{gsub(/ /,""); print "| " b " | " $4 " | " $7 " | " $9 " |"}' >> $OUT
  q "final bb32k $b: $(grep -E '^\| *32768 ' $B/$TAG-bb9-$b.md | awk -F'|' '{gsub(/ /,""); printf "%s:%s ", $4, $9}')"
done
{ echo; echo "| build | spec | w1 gen t/s | w2 gen t/s | accepted / drafted |"; echo "|---|---|---:|---:|---|"; } >> $OUT
for b in final prod; do P=$(pfx $b); for spec in none mtp3; do SPEC=""; [ $spec = mtp3 ] && SPEC="--spec-type draft-mtp --spec-draft-n-max 3"; LOG=$B/$TAG-mtp9-$b-$spec-srv.log
  LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M $D4 -fa on -b 2048 -ub 2048 -cb --host 127.0.0.1 --port 8089 -np 1 -c 34816 $SPEC > $LOG 2>&1 & S=$!
  if wait_ready 8089 $S; then C=$B/$TAG-mtp9-$b-$spec-client.md
    python3 $B/mtp-depth-client.py http://127.0.0.1:8089 "$TAG-mtp9-$b-$spec" $LOG --context-tokens 32768 --conc 1 --gen 300 --waves 2 --warmup 300 > $C 2>>$B/$TAG-mtp.err
    w1=$(grep 'wave 1' $C | tail -1); w2=$(grep 'wave 2' $C | tail -1)
    g1=$(echo "$w1" | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}'); g2=$(echo "$w2" | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}'); acc=$(echo "$w2" | awk -F'|' '{gsub(/^ +| +$/,"",$11); print $11}')
    echo "| $b | $spec | ${g1:--} | ${g2:--} | ${acc:--} |" >> $OUT; q "final mtp $b $spec: g2 $g2 acc $acc"
  else q "final mtp $b $spec: SERVER FAILED"; fi
  stop_server $S
done; done
LD_LIBRARY_PATH=$F/lib timeout 900 $F/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f /root/models/wikitext-2-raw/wiki.test.raw > $B/$TAG-ppl-final.log 2>&1
echo "- final perplexity 16K/6: $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-ppl-final.log | tail -1)" >> $OUT; q "final ppl: $(grep -oE 'PPL = [0-9.]+' $B/$TAG-ppl-final.log | tail -1)"
echo "# final done $(date -Is)" >> $OUT; kill $SAMP 2>/dev/null; restore; q "night0909 final ALLDONE"; touch $B/.final-done
