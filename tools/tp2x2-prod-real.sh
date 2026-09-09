#!/bin/bash
# BENCHMARKS-TODO item 8: two tensor-split pairs on the production build. C1: -np 8 each (as the stock run: 75.2 / 79.4 at 8 / 16 clients);
# C2: -np 16 each (untested), dual-server-bench 1300-token prompts, 256 gen.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-mxxmfh-tp2x2; P=/opt/llama.cpp-mxxm-fh
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; pkill -f "^/opt/llama.cpp[^ ]*/bin/llama-server .*--port 80(89|90)" 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib
OUT=$B/$TAG.md; echo "# $TAG  $(date -Is)  two tp2 servers on the production build (8089 = rocm0,rocm1 / 8090 = rocm2,rocm3); libggml-hip: $(ldd $P/bin/llama-server | grep -o '/opt/[^ ]*libggml-hip[^ ]*')" > $OUT
srv() { local port=$1; shift; $P/bin/llama-server -m $M -fa on -b 2048 -ub 2048 -cb --host 127.0.0.1 --port $port "$@" > $B/$TAG-server-$port.log 2>&1 & echo $!; }
stage() { local name=$1 np=$2 ctx=$3 conc=$4
  { echo; echo "## $name. two tp2 servers -np $np -c $ctx each; dual-server-bench conc $conc total, 1300-token prompts, 256 gen"; } >> $OUT
  local S1 S2; S1=$(srv 8089 --device rocm0,rocm1 -sm tensor -np $np -c $ctx); S2=$(srv 8090 --device rocm2,rocm3 -sm tensor -np $np -c $ctx)
  if wait_server 8089 $S1 && wait_server 8090 $S2; then python3 $B/dual-server-bench.py $TAG-$name --urls http://127.0.0.1:8089,http://127.0.0.1:8090 --conc $conc --gen 256 --prompt-tokens 1300 >> $OUT 2>>$B/$TAG-$name.err; else echo "server failed" >> $OUT; fi
  stop_server $S1 $S2; log "$name done: $(grep -E '^\| [0-9]+ ' $OUT | tail -2 | cut -c1-100 | tr '\n' ' ')"; }
stage C1 8 262144 8,16
stage C2 16 131072 16,32
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
