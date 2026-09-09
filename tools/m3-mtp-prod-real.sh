#!/bin/bash
# NEXT-STEPS M3: MTP on the production build: 4 slots x draft 1/2/3 and 8 slots x draft 1 (verify batch 16 = the new MMVQ width), 2K and 32K prompts.
# Same protocol as mtp-depth-sweep2.sh (two waves, wave 2 = clean decode factor), server = /opt/llama.cpp-mxxm-fh.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-mxxmfh-m3-mtp; P=/opt/llama.cpp-mxxm-fh
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; pkill -f "^/opt/llama.cpp[^ ]*/bin/llama-server .*--port 8089" 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib
OUT=$B/$TAG.md
{ echo "# $TAG  $(date -Is)  production build tp4, greedy, 300 generated, N concurrent distinct wikitext prompts, two waves (wave 2 = clean decode); libggml-hip: $(ldd $P/bin/llama-server | grep -o '/opt/[^ ]*libggml-hip[^ ]*')"
  echo "| variant | depth | conc | wall s | prompt tok | TTFT mean / max s | per-req gen t/s | sum per-req gen t/s | wave agg gen t/s | accepted / drafted |"; echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"; } > $OUT
run() { local name=$1 depth=$2 np=$3; shift 3; local LOG=$B/$TAG-$name-$depth-np$np.log
  $P/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -b 2048 -ub 2048 -cb --host 127.0.0.1 --port 8089 -np $np -c $((np*(depth+2048))) "$@" > $LOG 2>&1 &
  local S=$!
  if wait_server 8089 $S; then python3 $B/mtp-depth-client.py http://127.0.0.1:8089 "$name" $LOG --context-tokens $depth --conc $np --gen 300 --waves 2 >> $OUT 2>>$B/$TAG-$name.err || echo "| $name | $depth | $np | client failed | | | | | |" >> $OUT
  else echo "| $name | $depth | $np | server failed: $(grep -ihE 'error|assert' $LOG | head -1 | cut -c1-80) | | | | | |" >> $OUT; fi
  stop_server $S; log "$name @$depth np$np: $(tail -1 $OUT | cut -c1-150)"; }
for depth in 2048 32768; do
  run none   $depth 4
  run mtp-n1 $depth 4 --spec-type draft-mtp --spec-draft-n-max 1
  run mtp-n2 $depth 4 --spec-type draft-mtp --spec-draft-n-max 2
  run mtp-n3 $depth 4 --spec-type draft-mtp --spec-draft-n-max 3
  run none   $depth 8
  run mtp-n1 $depth 8 --spec-type draft-mtp --spec-draft-n-max 1
done
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
