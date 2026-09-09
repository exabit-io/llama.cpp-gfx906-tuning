#!/bin/bash
# TODO 10/11: server level on the production build (+custom AR env): team (np16, conc 4/8/12/16), busy (np32, conc 16/32), pairs (2 x tp2 np8, conc 8/16);
# the 2026-09-07 production at np32 (conc 16/32) as the reference for TODO 10.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-server-final; P=/opt/llama.cpp-prod; P0=/opt/llama.cpp-mxxm-fh
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; pkill -f "^/opt/llama.cpp[^ ]*/bin/llama-server .*--port 80(89|90)" 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
AR="GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481"
OUT=$B/$TAG.md; echo "# $TAG  $(date -Is)  server level, 1300-token prompts / 256 generated" > $OUT
one() { local P=$1 name=$2 np=$3 ctx=$4 conc=$5; shift 5; local LOG=$B/$TAG-$name.log
  env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np $np -cb -c $ctx -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 > $LOG 2>&1 &
  local S=$!; { echo; echo "## $name: $P -np $np -c $ctx, conc $conc"; } >> $OUT
  if wait_server 8089 $S; then python3 $B/server-bench.py http://127.0.0.1:8089 $TAG-$name --conc $conc --gen 256 --prompt-tokens 1300 >> $OUT 2>>$LOG; else echo "server failed" >> $OUT; fi
  stop_server $S; log "$name: $(grep -E '^\| [0-9]+ ' $OUT | tail -2 | awk -F'|' '{printf "conc%s agg %s; ", $2, $5}')"; }
pairs() { local P=$1 name=$2 np=$3 ctx=$4 conc=$5; shift 5
  env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M --device rocm0,rocm1 -sm tensor -fa on -np $np -cb -c $ctx -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 > $B/$TAG-$name-0.log 2>&1 & local S1=$!
  env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-server -m $M --device rocm2,rocm3 -sm tensor -fa on -np $np -cb -c $ctx -b 2048 -ub 2048 --host 127.0.0.1 --port 8090 > $B/$TAG-$name-1.log 2>&1 & local S2=$!
  { echo; echo "## $name: two tp2 servers -np $np -c $ctx, conc $conc total"; } >> $OUT
  if wait_server 8089 $S1 && wait_server 8090 $S2; then python3 $B/dual-server-bench.py $TAG-$name --urls http://127.0.0.1:8089,http://127.0.0.1:8090 --conc $conc --gen 256 --prompt-tokens 1300 >> $OUT 2>>$B/$TAG-$name.err; else echo "server failed" >> $OUT; fi
  stop_server $S1 $S2; log "$name: $(grep -E '^\| [0-9]+ ' $OUT | tail -2 | cut -c1-80 | tr '\n' ' ')"; }
one $P team16 16 $((16*32768)) 4,8,12,16 $AR
one $P busy32 32 $((32*32768)) 16,32 $AR
pairs $P pairs8 8 262144 8,16 $AR
one $P0 prod0907-np32 32 $((32*32768)) 16,32 X=1
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
