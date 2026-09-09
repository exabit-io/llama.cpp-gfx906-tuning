#!/bin/bash
# post28 (review follow-up, 2026-09-08 evening): A. MTP promotion gate (prod / branch repack / branch --no-repack, drafting off and draft 3,
# 2K and 32K, three interleaved rounds with rotated order)  B. width 6/7/8/9 isolation with repeated 8-cells (--no-repack vs production, x3)
# C. cold start: four fresh processes per build, load-to-ready, first-request TTFT, decode warm-up curve, first use of the batch-8 shape
# D. Flash-Next per-sample tg128 x8 (repack on/off) and perplexity on a held-out corpus (our own 2026-09 docs) vs wikitext, 27B for reference
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
PR=/opt/llama.cpp-prod; R2=/opt/llama.cpp-gfx906-master-r2; TAG=qwen38-27b-post28
q() { echo "$(date -Is) $*" >> $Q; }
q "=== post28 start"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/post28-clocks.txt
QUEUE="^/bin/bash $B/post28[.]sh" DONEFLAG=$B/.post28-done SMCLOG=$B/smc-power-post28.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-post28.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-post28.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"
wait_ready() { local port=$1 pid=$2 i; for i in $(seq 1 900); do curl -sf http://127.0.0.1:$port/health >/dev/null && return 0; kill -0 $pid 2>/dev/null || return 1; sleep 1; done; return 1; }
prefix() { case $1 in prod) echo $PR;; r2-*) echo $R2;; esac; }
extra() { case $1 in r2-nr1) echo "--no-repack";; *) echo "";; esac; }
now() { date +%s.%N; }
# ---------------- A. MTP gate
OUT=$B/$TAG-mtp-gate.md
{ echo "# $TAG MTP gate $(date -Is): single user, -np 1 -c 34816, gen 300, wave 1 = prefill+decode, wave 2 = decode only (cache_prompt); three rounds, order rotated"
  echo "| round | build | spec | ctx | load s | w1 prefill t/s | w1 gen t/s | w2 gen t/s | accepted / drafted |"; echo "|---|---|---|---:|---:|---:|---:|---:|---|"; } > $OUT
ORDERS=("prod r2-nr1 r2-nr0" "r2-nr0 prod r2-nr1" "r2-nr1 r2-nr0 prod")
for r in 1 2 3; do for bld in ${ORDERS[$((r-1))]}; do for spec in none mtp3; do
  P=$(prefix $bld); X=$(extra $bld); SPEC=""; [ $spec = mtp3 ] && SPEC="--spec-type draft-mtp --spec-draft-n-max 3"
  LOG=$B/$TAG-mtp-$bld-$spec-r$r-srv.log; t0=$(now)
  LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M $D4 -fa on -b 2048 -ub 2048 -cb --host 127.0.0.1 --port 8089 -np 1 -c 34816 $SPEC $X > $LOG 2>&1 & S=$!
  if wait_ready 8089 $S; then load=$(echo "$(now) - $t0" | bc)
    for depth in 2048 32768; do C=$B/$TAG-mtp-$bld-$spec-r$r-$depth-client.md
      python3 $B/mtp-depth-client.py http://127.0.0.1:8089 "$TAG-mtp-$bld-$spec-r$r-$depth" $LOG --context-tokens $depth --conc 1 --gen 300 --waves 2 > $C 2>>$B/$TAG-mtp.err
      w1=$(grep 'wave 1' $C | tail -1); w2=$(grep 'wave 2' $C | tail -1)
      pp=$(echo "$w1" | awk -F'|' '{gsub(/ /,"",$6); split($7,a,"/"); gsub(/ /,"",a[1]); if (a[1]+0>0) printf "%.0f", $6/a[1]; else print "-"}')
      g1=$(echo "$w1" | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}'); g2=$(echo "$w2" | awk -F'|' '{gsub(/^ +| +$/,"",$8); print $8}'); acc=$(echo "$w2" | awk -F'|' '{gsub(/^ +| +$/,"",$11); print $11}')
      echo "| $r | $bld | $spec | $depth | $(printf %.1f $load) | ${pp:--} | ${g1:--} | ${g2:--} | ${acc:--} |" >> $OUT; q "post28 A r$r $bld $spec $depth: load $(printf %.1f $load) pp $pp g1 $g1 g2 $g2 acc $acc"
    done
  else echo "| $r | $bld | $spec | - | server failed | | | | |" >> $OUT; q "post28 A r$r $bld $spec: SERVER FAILED"; fi
  stop_server $S
done; done; done
echo "# done $(date -Is)" >> $OUT
# ---------------- B. width isolation
OUT=$B/$TAG-width8.md
{ echo "# $TAG width isolation $(date -Is): batched-bench tp4 2K, -npl 8,6,7,8,9,8,12,16,8 (the 8-cell first, after 7, after 9, last); branch --no-repack vs production, three interleaved rounds"
  echo "| round | build | 8 (first) | 6 | 7 | 8 | 9 | 8 | 12 | 16 | 8 (last) |"; echo "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"; } > $OUT
for r in 1 2 3; do if [ $r = 2 ]; then order="prod r2-nr1"; else order="r2-nr1 prod"; fi; for bld in $order; do P=$(prefix $bld); X=$(extra $bld)
  LD_LIBRARY_PATH=$P/lib timeout 1200 $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 8,6,7,8,9,8,12,16,8 $X > $B/$TAG-width8-$bld-r$r-bb.md 2>/dev/null
  cells=$(grep -E '^\|' $B/$TAG-width8-$bld-r$r-bb.md | grep -v 'PP \|---' | awk -F'|' '{gsub(/ /,"",$9); printf "%s | ", $9}'); echo "| $r | $bld | $cells" >> $OUT; q "post28 B r$r $bld: $cells"
done; done
echo "# done $(date -Is)" >> $OUT
# ---------------- C. cold start
OUT=$B/$TAG-cold-start.md
{ echo "# $TAG cold start $(date -Is): fresh llama-server (-np 8 -c 65536) x4 per build; load-to-ready, then the client's steps (seq1 = 2K prefill + 64 tokens, seq2-6 = decode-only 64-token continuations, batch8 = 8 x 1300-token prompts + 64 tokens)"
  echo "| process | step | s since ready | prompt n | prompt ms | gen n | gen t/s |"; echo "|---|---|---:|---:|---:|---:|---:|"; } > $OUT
ORD=("prod r2-nr0 r2-nr1" "r2-nr0 r2-nr1 prod" "r2-nr1 prod r2-nr0" "prod r2-nr1 r2-nr0")
for i in 1 2 3 4; do for bld in ${ORD[$((i-1))]}; do P=$(prefix $bld); X=$(extra $bld); LOG=$B/$TAG-cold-$bld-p$i-srv.log; t0=$(now)
  LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M $D4 -fa on -b 2048 -ub 2048 -cb --host 127.0.0.1 --port 8089 -np 8 -c 65536 $X > $LOG 2>&1 & S=$!
  if wait_ready 8089 $S; then ready=$(now); echo "| $bld-p$i | load-to-ready | $(echo "$ready - $t0" | bc | xargs printf %.1f) | | | | |" >> $OUT
    python3 $B/cold-start-client.py http://127.0.0.1:8089 "$bld-p$i" $ready >> $OUT 2>>$B/$TAG-cold.err; q "post28 C $bld-p$i: $(grep "$bld-p$i" $OUT | awk -F'|' '{gsub(/ /,"",$8); printf "%s ", $8}')"
  else echo "| $bld-p$i | server failed | | | | | |" >> $OUT; q "post28 C $bld-p$i: SERVER FAILED"; fi
  stop_server $S
done; done
echo "# done $(date -Is)" >> $OUT
# ---------------- D. Flash-Next warm-up curve + held-out perplexity
OUT=$B/qwen38-flash-next-2.md; MD=/root/models/Qwen3.8-Flash-Next-UD-Q4_K_XL; FM=$MD/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf; HO=/root/models/heldout-exabit-docs.txt; WT=/root/models/wikitext-2-raw/wiki.test.raw
LM=""; $R2/bin/llama-bench --help 2>&1 | grep -q -- '--load-mode' && LM="--load-mode dio"
{ echo "# qwen38-flash-next-2 $(date -Is): $R2, four dies -sm tensor, LLAMA_PLE_SHARD=1 ($LM)"; echo; echo "## per-sample tg128 x8 (llama-bench -p 0 -n 128 -r 8), repack on / off"; echo "| run | mean ± sd | samples |"; echo "|---|---:|---|"; } > $OUT
for nr in 0 1; do LLAMA_PLE_SHARD=1 LD_LIBRARY_PATH=$R2/lib timeout 3600 $R2/bin/llama-bench -m $FM --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 0 -n 128 -r 8 -nr $nr $LM -o json > $B/qwen38-flash-next-2-tg-nr$nr.json 2>$B/qwen38-flash-next-2-tg-nr$nr.err
  python3 - $B/qwen38-flash-next-2-tg-nr$nr.json "repack $([ $nr = 0 ] && echo on || echo off)" >> $OUT <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))[0]; s = [128 / (x / 1e9) for x in d['samples_ns']]
    print(f"| {sys.argv[2]} | {d['avg_ts']:.1f} ± {d['stddev_ts']:.1f} | " + ' '.join(f"{x:.1f}" for x in s) + " |")
except Exception as e: print(f"| {sys.argv[2]} | fail | {e} |")
PY
  q "post28 D flash-next tg nr$nr: $(tail -1 $OUT)"; done
{ echo; echo "## perplexity -c 2048 --chunks 8: wikitext-2 test vs held-out corpus (our own 2026-09 docs, $(wc -c < $HO) bytes)"; echo "| model | wikitext | held-out |"; echo "|---|---:|---:|"; } >> $OUT
ppl() { grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $1 | tail -1 | sed 's/Final estimate: PPL = //'; }
LLAMA_PLE_SHARD=1 LD_LIBRARY_PATH=$R2/lib timeout 3600 $R2/bin/llama-perplexity -m $FM $D4 -fa on -b 2048 -ub 2048 -c 2048 --chunks 8 $LM -f $WT > $B/qwen38-flash-next-2-ppl-wt.log 2>&1
LLAMA_PLE_SHARD=1 LD_LIBRARY_PATH=$R2/lib timeout 3600 $R2/bin/llama-perplexity -m $FM $D4 -fa on -b 2048 -ub 2048 -c 2048 --chunks 8 $LM -f $HO > $B/qwen38-flash-next-2-ppl-ho.log 2>&1
echo "| Flash-Next UD-Q4_K_XL (branch, PLE sharded) | $(ppl $B/qwen38-flash-next-2-ppl-wt.log) | $(ppl $B/qwen38-flash-next-2-ppl-ho.log) |" >> $OUT; q "post28 D flash-next ppl: $(tail -1 $OUT)"
LD_LIBRARY_PATH=$PR/lib timeout 1800 $PR/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 2048 --chunks 8 -f $WT > $B/$TAG-27b-ppl-wt.log 2>&1
LD_LIBRARY_PATH=$PR/lib timeout 1800 $PR/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 2048 --chunks 8 -f $HO > $B/$TAG-27b-ppl-ho.log 2>&1
echo "| 27B Q8_0 (production) | $(ppl $B/$TAG-27b-ppl-wt.log) | $(ppl $B/$TAG-27b-ppl-ho.log) |" >> $OUT; q "post28 D 27b ppl: $(tail -1 $OUT)"
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; for c in 0 1; do echo 413000000 > $RAPL/constraint_${c}_power_limit_uw; done; pkill -f "smc-log[.]sh"; touch $B/.post28-done; q "=== post28 done"
