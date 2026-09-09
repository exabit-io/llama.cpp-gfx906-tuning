#!/bin/bash
# s1b-test (after post28 and the builds): S1b candidate 1 (repacked nc mat-vec through 16 columns) a = as written, a2 = + launch bounds,
# against the branch r2 (repack on) and production; and the S1 upstream patch (tile table alone on master 5d806aa25) against pristine master.
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress; TAG=qwen38-27b-s1b
PR=/opt/llama.cpp-prod; R2=/opt/llama.cpp-gfx906-master-r2; A=/opt/llama.cpp-gfx906-s1b-a; A2=/opt/llama.cpp-gfx906-s1b-a2; A3=/opt/llama.cpp-gfx906-s1b-a3; S1=/opt/llama.cpp-s1-upstream; UP=/opt/llama.cpp-master
q() { echo "$(date -Is) $*" >> $Q; }
while [ ! -f $B/.post28-done ] || ! grep -q "BUILD-S1B-A3-OK\|BUILD-S1B-A3-FAIL" $B/build-s1b-a3.log 2>/dev/null; do sleep 30; done
q "=== s1b-test start"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/s1b-test-clocks.txt
QUEUE="^/bin/bash $B/s1b-test[.]sh" DONEFLAG=$B/.s1b-test-done SMCLOG=$B/smc-power-s1b-test.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-s1b-test.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-s1b-test.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md; F=/root/models/wikitext-2-raw/wiki.test.raw
BUILDS="r2 a a2 a3 prod"; [ -x $A3/bin/llama-server ] || BUILDS="r2 a a2 prod"
pfx() { case $1 in prod) echo $PR;; r2) echo $R2;; a) echo $A;; a2) echo $A2;; a3) echo $A3;; s1) echo $S1;; up) echo $UP;; esac; }
# A0. MTP gate, warmed: 600 generated tokens after load before the measured waves (post28 A measured inside the warm-up window)
OUT=$B/$TAG-mtp-gate-warm.md
{ echo "# $TAG MTP gate (warmed) $(date -Is): as post28 A but each server generates 600 tokens first; wave 2 = decode only"
  echo "| round | build | spec | ctx | w1 gen t/s | w2 gen t/s | accepted / drafted |"; echo "|---|---|---|---:|---:|---:|---|"; } > $OUT
wait_ready() { local port=$1 pid=$2 i; for i in $(seq 1 900); do curl -sf http://127.0.0.1:$port/health >/dev/null && return 0; kill -0 $pid 2>/dev/null || return 1; sleep 1; done; return 1; }
ORDERS=("prod r2-nr1 r2-nr0" "r2-nr0 prod r2-nr1" "r2-nr1 r2-nr0 prod")
for r in 1 2 3; do for bld in ${ORDERS[$((r-1))]}; do for spec in none mtp3; do
  case $bld in prod) P=$PR; X="";; r2-nr1) P=$R2; X="--no-repack";; r2-nr0) P=$R2; X="";; esac; SPEC=""; [ $spec = mtp3 ] && SPEC="--spec-type draft-mtp --spec-draft-n-max 3"
  LOG=$B/$TAG-mtpw-$bld-$spec-r$r-srv.log
  LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M $D4 -fa on -b 2048 -ub 2048 -cb --host 127.0.0.1 --port 8089 -np 1 -c 34816 $SPEC $X > $LOG 2>&1 & S=$!
  if wait_ready 8089 $S; then for depth in 2048 32768; do C=$B/$TAG-mtpw-$bld-$spec-r$r-$depth-client.md
      python3 $B/mtp-depth-client.py http://127.0.0.1:8089 "$TAG-mtpw-$bld-$spec-r$r-$depth" $LOG --context-tokens $depth --conc 1 --gen 300 --waves 2 --warmup $([ $depth = 2048 ] && echo 600 || echo 0) > $C 2>>$B/$TAG-mtpw.err
      w1=$(grep 'wave 1' $C | tail -1); w2=$(grep 'wave 2' $C | tail -1)
      g1=$(echo "$w1" | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}'); g2=$(echo "$w2" | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}'); acc=$(echo "$w2" | awk -F'|' '{gsub(/^ +| +$/,"",$11); print $11}')
      echo "| $r | $bld | $spec | $depth | ${g1:--} | ${g2:--} | ${acc:--} |" >> $OUT; q "s1b-test A0 r$r $bld $spec $depth: g1 $g1 g2 $g2 acc $acc"; done
  else echo "| $r | $bld | $spec | - | server failed | | |" >> $OUT; q "s1b-test A0 r$r $bld $spec: SERVER FAILED"; fi
  stop_server $S
done; done; done
echo "# done $(date -Is)" >> $OUT
OUT=$B/$TAG.md
echo "# $TAG $(date -Is): S1b candidate 1 (a = nc mat-vec 9-16 cols at 64 VGPRs, spills from 11; a2 = bare launch bound, 129-228 VGPRs; a3 = 64 VGPRs through 10 cols, 128 above) vs branch r2 and production; S1 tile-table patch vs pristine master" > $OUT
# A. correctness: test-backend-ops MUL_MAT on the a2 (or a) build and the S1 build
TBO=/root/exabit-llama.cpp-s1b/build/bin/test-backend-ops; P=$A3; [ -x $A3/bin/llama-server ] || P=$A2
LD_LIBRARY_PATH=/root/exabit-llama.cpp-s1b/build/bin:$P/lib timeout 2400 $TBO test -b ROCm0 -o MUL_MAT > $B/$TAG-tbo-mulmat.log 2>&1; r1=$(grep -E 'tests passed|tests failed' $B/$TAG-tbo-mulmat.log | tail -1)
TBO1=/root/exabit-llama.cpp-s1/build/bin/test-backend-ops; LD_LIBRARY_PATH=/root/exabit-llama.cpp-s1/build/bin:$S1/lib timeout 2400 $TBO1 test -b ROCm0 -o MUL_MAT > $B/$TAG-s1-tbo-mulmat.log 2>&1; r2=$(grep -E 'tests passed|tests failed' $B/$TAG-s1-tbo-mulmat.log | tail -1)
{ echo; echo "## A. test-backend-ops MUL_MAT: S1b build: ${r1:-?}; S1 build: ${r2:-?}"; } >> $OUT; q "s1b-test tbo: s1b ${r1:-?} / s1 ${r2:-?}"
# B. batched widths, tp4 2K, two interleaved rounds, repack ON on the branch builds
{ echo; echo "## B. batched-bench tp4 2K decode tok/s, -npl 7,8,9,12,16,17,24,32 (repack on for r2/a/a2)"; echo "| round | build | 7 | 8 | 9 | 12 | 16 | 17 | 24 | 32 |"; echo "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"; } >> $OUT
for r in 1 2; do for b in $BUILDS; do P=$(pfx $b)
  LD_LIBRARY_PATH=$P/lib timeout 1200 $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 7,8,9,12,16,17,24,32 > $B/$TAG-B-$b-r$r-bb.md 2>/dev/null
  cells=$(grep -E '^\|' $B/$TAG-B-$b-r$r-bb.md | grep -v 'PP \|---' | awk -F'|' '{gsub(/ /,"",$9); printf "%s | ", $9}'); echo "| $r | $b | $cells" >> $OUT; q "s1b-test B r$r $b: $cells"; done; done
# B2. shape transitions in one process (post28 B: an 8-cell after the 16-cell reads -6% on the branch, every other 8-cell is at parity)
{ echo; echo "## B2. batched-bench tp4 2K, -npl 16,8,16,8,12,8,4,8,1,8,32,8: does a narrower batch after a wider one pay on the branch (repack off / on), the S1b build, production?"; echo "| build | 16 | 8 | 16 | 8 | 12 | 8 | 4 | 8 | 1 | 8 | 32 | 8 |"; echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"; } >> $OUT
for b in r2-nr1 r2 a3 prod; do case $b in r2-nr1) P=$R2; X="--no-repack";; *) P=$(pfx $b); X="";; esac; [ -x $P/bin/llama-batched-bench ] || continue
  LD_LIBRARY_PATH=$P/lib timeout 1500 $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 16,8,16,8,12,8,4,8,1,8,32,8 $X > $B/$TAG-B2-$b-bb.md 2>/dev/null
  cells=$(grep -E '^\|' $B/$TAG-B2-$b-bb.md | grep -v 'PP \|---' | awk -F'|' '{gsub(/ /,"",$9); printf "%s | ", $9}'); echo "| $b | $cells" >> $OUT; q "s1b-test B2 $b: $cells"; done
# C. one die, 512 context
{ echo; echo "## C. batched-bench rocm0 512, -npl 4,8,12,16"; echo "| build | 4 | 8 | 12 | 16 |"; echo "|---|---:|---:|---:|---:|"; } >> $OUT
for b in $BUILDS; do P=$(pfx $b)
  LD_LIBRARY_PATH=$P/lib timeout 900 $P/bin/llama-batched-bench -m $M --device rocm0 -fa on -b 2048 -ub 2048 -c 9216 -npp 512 -ntg 128 -npl 4,8,12,16 > $B/$TAG-C-$b-bb.md 2>/dev/null
  cells=$(grep -E '^\|' $B/$TAG-C-$b-bb.md | grep -v 'PP \|---' | awk -F'|' '{gsub(/ /,"",$9); printf "%s | ", $9}'); echo "| $b | $cells" >> $OUT; q "s1b-test C $b: $cells"; done
# D. llama-bench pp2048/tg128 (prefill must survive) + ppl 16K on the S1b build
{ echo; echo "## D. llama-bench -p 2048 -n 128 -r 3 tp4 + rocm0"; echo "| build | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |"; echo "|---|---:|---:|---:|---:|"; } >> $OUT
lb() { local P=$1 name=$2; shift 2; env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3,rocm0 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 128 -d 0 -r 3 -o md > $B/$TAG-$name-lb.md 2>$B/$TAG-$name-lb.err
  local cells; cells=$(grep -E '^\| qwen' $B/$TAG-$name-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$12); printf "%s | ", $12}'); echo "| $name | $cells" >> $OUT; q "s1b-test lb $name: $cells"; }
for b in $BUILDS; do lb $(pfx $b) $b X=1; done
P=$A3; [ -x $A3/bin/llama-server ] || P=$A2
LD_LIBRARY_PATH=$P/lib $P/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $B/$TAG-ppl.log 2>&1
fe=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-ppl.log | tail -1); { echo; echo "## D2. perplexity 16K chunks 6, S1b build: ${fe:-none} (r2 5.6173, production 5.5969)"; } >> $OUT; q "s1b-test ppl ${fe:-none}"
# E. server level team16 conc 8,16: S1b build (repack on) vs production, one round each, S1b first
{ echo; echo "## E. server level team16, conc 8,16 (server-bench.py 1300/256)"; } >> $OUT
for b in a3 prod; do P=$(pfx $b); [ -x $P/bin/llama-server ] || { P=$A2; b=a2; }
  LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M $D4 -fa on -np 16 -cb -c $((16*32768)) -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 > $B/$TAG-E-$b-srv.log 2>&1 & S=$!
  if wait_server 8089 $S; then { echo; echo "### $b"; } >> $OUT; python3 $B/server-bench.py http://127.0.0.1:8089 $TAG-E-$b --conc 8,16 --gen 256 --prompt-tokens 1300 >> $OUT 2>>$B/$TAG-E-$b-srv.log; fi
  stop_server $S; q "s1b-test srv $b: $(grep -E '^\| [0-9]+ ' $OUT | tail -2 | awk -F'|' '{printf "c%s %s; ", $2, $5}')"; done
# F. S1 upstream patch vs pristine master: llama-bench, batched 8/16/32, ppl
{ echo; echo "## F. S1 tile-table patch on master 5d806aa25 vs pristine master: llama-bench -r 3"; echo "| build | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |"; echo "|---|---:|---:|---:|---:|"; } >> $OUT
lb $UP up X=1; lb $S1 s1 X=1; lb $UP up-repeat X=1; lb $S1 s1-repeat X=1
{ echo; echo "### batched tp4 2K -npl 1,8,16,32"; echo "| build | 1 | 8 | 16 | 32 |"; echo "|---|---:|---:|---:|---:|"; } >> $OUT
for b in up s1; do P=$(pfx $b); LD_LIBRARY_PATH=$P/lib timeout 1200 $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 1,8,16,32 > $B/$TAG-F-$b-bb.md 2>/dev/null
  cells=$(grep -E '^\|' $B/$TAG-F-$b-bb.md | grep -v 'PP \|---' | awk -F'|' '{gsub(/ /,"",$9); printf "%s | ", $9}'); echo "| $b | $cells" >> $OUT; q "s1b-test F $b: $cells"; done
for b in up s1; do P=$(pfx $b); LD_LIBRARY_PATH=$P/lib $P/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $B/$TAG-F-$b-ppl.log 2>&1
  fe=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-F-$b-ppl.log | tail -1); echo "- ppl 16K $b: ${fe:-none}" >> $OUT; q "s1b-test F ppl $b ${fe:-none}"; done
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; for c in 0 1; do echo 413000000 > $RAPL/constraint_${c}_power_limit_uw; done; pkill -f "smc-log[.]sh"; touch $B/.s1b-test-done; q "=== s1b-test done"
