#!/bin/bash
# Final combined configuration (2026-09-08): fusion build (v8 one-column load, add+norm fusion, q8 fusion, GDN fold per .gdn-mask) with the
# adopted custom-AR settings (and GGML_CUDA_GRAPH_OPT=2 if .graphopt says so) against the 2026-09-07 production build.
# A. llama-bench pp2048/tg128 tp4 + rocm0 (-r 3); B. batched 1,2,4,8,16,32 at 2K; C. single-user MTP server (np 1, draft 3) at 2K and 32K; D. perplexity.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-final-config; NQ=/opt/llama.cpp-mxxm-fh-nq; PR=/opt/llama.cpp-mxxm-fh
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; pkill -f "^/opt/llama.cpp[^ ]*/bin/llama-server .*--port 8089" 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
MASK=$(cat $B/.gdn-mask 2>/dev/null || echo 0); GO=$(cat $B/.graphopt 2>/dev/null || echo 0)
FINAL="GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481 GGML_CUDA_GDN_PREFUSE=$MASK"; [ "$GO" = 2 ] && FINAL="$FINAL GGML_CUDA_GRAPH_OPT=2"
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md; F=/root/models/wikitext-2-raw/wiki.test.raw
{ echo "# $TAG  $(date -Is)  final = $NQ with: $FINAL  vs production $PR (libggml-hip: $(ldd $NQ/bin/llama-bench | grep -o '/opt/[^ ]*libggml-hip[^ ]*'))"; } > $OUT
# A0. decode-path numerics: greedy text and a KL divergence at 64-token batches (the folds apply at <= 64 rows; plain perplexity runs 2048-token batches and never exercises them)
PROMPT="The four dies of a Radeon Pro Vega II Duo share one XGMI ring. In a tensor split, each decode token"
LD_LIBRARY_PATH=$PR/lib $PR/bin/llama-completion -m $M $D4 -fa on -b 2048 -ub 2048 -c 4096 -n 200 --temp 0 --seed 1 -no-cnv -p "$PROMPT" > $B/$TAG-prod-greedy.txt 2>/dev/null
env $FINAL LD_LIBRARY_PATH=$NQ/lib $NQ/bin/llama-completion -m $M $D4 -fa on -b 2048 -ub 2048 -c 4096 -n 200 --temp 0 --seed 1 -no-cnv -p "$PROMPT" > $B/$TAG-final-greedy.txt 2>/dev/null
pos=$(cmp $B/$TAG-final-greedy.txt $B/$TAG-prod-greedy.txt 2>/dev/null | grep -oE 'byte [0-9]+' | grep -oE '[0-9]+'); [ -z "$pos" ] && pos=-1
LD_LIBRARY_PATH=$PR/lib $PR/bin/llama-perplexity -m $M $D4 -fa on -b 64 -ub 64 -c 2048 --chunks 8 -f $F --kl-divergence-base $B/$TAG-base.kld > $B/$TAG-kld-base.log 2>&1
env $FINAL LD_LIBRARY_PATH=$NQ/lib $NQ/bin/llama-perplexity -m $M $D4 -fa on -b 64 -ub 64 -c 2048 --chunks 8 -f $F --kl-divergence-base $B/$TAG-base.kld --kl-divergence > $B/$TAG-kld-final.log 2>&1
kl=$(grep -E 'Mean\s+KLD|Maximum KLD|Same top p|Mean.*ln\(PPL' $B/$TAG-kld-final.log | tr -s ' ' | tr '\n' ';' | cut -c1-300)
{ echo; echo "## A0. decode-path numerics vs production: greedy 200 tokens differ at char ${pos} (-1 = identical); KL divergence at 64-token batches over 8 x 2048 tokens: ${kl:-none}"; } >> $OUT; log "A0: greedy $pos; kl: ${kl:-none}"
{ echo; echo "## A. llama-bench -p 2048 -n 128 -r 3"; echo "| build | env | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |"; echo "|---|---|---:|---:|---:|---:|"; } >> $OUT
lb() { local P=$1 name=$2; shift 2; env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3,rocm0 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 128 -d 0 -r 3 -o md > $B/$TAG-$name-lb.md 2>$B/$TAG-$name-lb.err
  local cells; cells=$(grep -E '^\| qwen' $B/$TAG-$name-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$12); printf "%s | ", $12}'); echo "| $name | $* | $cells" >> $OUT; log "lb $name: $cells"; }
lb $PR prod BENCH=prod; lb $NQ final $FINAL; lb $NQ final-noar GGML_CUDA_GDN_PREFUSE=$MASK; lb $PR prod-repeat BENCH=prod2; lb $NQ final-repeat $FINAL
{ echo; echo "## B. batched-bench tp4 -npp 2048 -ntg 128 -npl 1,2,4,8,16,32 (decode tok/s)"; echo "| build | 1 | 2 | 4 | 8 | 16 | 32 |"; echo "|---|---:|---:|---:|---:|---:|---:|"; } >> $OUT
bb() { local P=$1 name=$2; shift 2; env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 1,2,4,8,16,32 > $B/$TAG-$name-bb.md 2>$B/$TAG-$name-bb.err
  local cells; cells=$(for n in 1 2 4 8 16 32; do grep -E '^\| *2048 ' $B/$TAG-$name-bb.md | awk -F'|' -v n=$n '$4+0==n {gsub(/ /,"",$9); printf "%s", $9}'; printf ' | '; done); echo "| $name | $cells" >> $OUT; log "bb $name: $cells"; }
bb $PR prod BENCH=prod; bb $NQ final $FINAL
{ echo; echo "## C. single user with MTP draft 3: llama-server -np 1, mtp-depth-client conc 1, 300 generated, two waves (wave 2 = decode only)"; echo "| build | depth | wall s | per-req gen t/s (wave 2) | accepted / drafted |"; echo "|---|---:|---:|---:|---|"; } >> $OUT
srv() { local P=$1 name=$2 depth=$3; shift 3; local LOG=$B/$TAG-$name-$depth-srv.log
  env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M $D4 -fa on -b 2048 -ub 2048 -cb --host 127.0.0.1 --port 8089 -np 1 -c $((depth+2048)) --spec-type draft-mtp --spec-draft-n-max 3 > $LOG 2>&1 &
  local S=$!
  if wait_server 8089 $S; then python3 $B/mtp-depth-client.py http://127.0.0.1:8089 "$name" $LOG --context-tokens $depth --conc 1 --gen 300 --waves 2 > $B/$TAG-$name-$depth-client.md 2>>$B/$TAG-$name.err
    local w2; w2=$(grep 'wave 2' $B/$TAG-$name-$depth-client.md | tail -1); echo "| $name | $depth | $(echo "$w2" | awk -F'|' '{gsub(/ /,"",$5); gsub(/^ +| +$/,"",$8); gsub(/^ +| +$/,"",$11); printf "%s | %s | %s |", $5, $8, $11}')" >> $OUT; log "srv $name @$depth: $(echo "$w2" | cut -c1-120)"
  else echo "| $name | $depth | server failed | | |" >> $OUT; fi
  stop_server $S; }
for depth in 2048 32768; do srv $PR prod $depth BENCH=prod; srv $NQ final $depth $FINAL; done
env $FINAL LD_LIBRARY_PATH=$NQ/lib $NQ/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $B/$TAG-final-ppl.log 2>&1
fe=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-final-ppl.log | tail -1); { echo; echo "## D. perplexity of the final configuration: ${fe:-none} (production 5.5969 +/- 0.062)"; } >> $OUT; log "ppl final: ${fe:-none}"
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
