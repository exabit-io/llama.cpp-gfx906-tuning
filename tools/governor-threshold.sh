#!/bin/bash
# M7 (bounded): the DC total against the host CPU power cap while the four dies serve 16 clients, and what an all-core host load costs
# serving at the 150 W cap. Host load = one `yes` per core under RAPL; the cap steps 150 -> 175 -> 200 -> 215 W only while the last
# SMC DC reading is below 1150 W; the script stops itself at 1180 W (the queue watchdog also kills at >= 1200 W twice).
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-governor; P=/opt/llama.cpp-prod; RAPL=/sys/class/powercap/intel-rapl:0
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
setrapl() { local c; for c in 0 1; do echo $(($1*1000000)) > $RAPL/constraint_${c}_power_limit_uw; done; }
stopload() { pkill -x yes 2>/dev/null; }
cleanup() { trap - INT TERM; stopload; setrapl 150; pkill -P $$ 2>/dev/null; pkill -f "^/opt/llama.cpp[^ ]*/bin/llama-server .*--port 8089" 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481
OUT=$B/$TAG.md; SMC=$B/$TAG-smc.log; nohup $B/smc-log.sh $SMC > /dev/null 2>&1 & SMCPID=$!
dc() { tail -1 $SMC 2>/dev/null | sed -n 's/.*PZ0G=\([0-9]*\).*/\1/p'; }
echo "# $TAG  $(date -Is)  16-client serving (tp4 -np 16, production build) beside an all-core host load at stepped RAPL caps" > $OUT
$P/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 16 -cb -c $((16*32768)) -b 2048 -ub 2048 --host 127.0.0.1 --port 8089 > $B/$TAG-server.log 2>&1 & S=$!
wait_server 8089 $S || { log "server failed"; kill $SMCPID; cleanup; }
{ echo; echo "## A. serving cost of host load at the 150 W cap: server-bench conc 16, host idle vs all-core load"; } >> $OUT
python3 $B/server-bench.py http://127.0.0.1:8089 $TAG-idle --conc 16 --gen 256 --prompt-tokens 1300 >> $OUT 2>/dev/null; log "idle host: $(grep -E '^\| 16 ' $OUT | tail -1 | cut -c1-100); DC $(dc) W"
for i in $(seq 1 $(nproc)); do yes > /dev/null & done; sleep 5; log "host load on at 150 W, DC $(dc) W"
python3 $B/server-bench.py http://127.0.0.1:8089 $TAG-load150 --conc 16 --gen 256 --prompt-tokens 1300 >> $OUT 2>/dev/null; log "loaded host 150 W: $(grep -E '^\| 16 ' $OUT | tail -1 | cut -c1-100); DC $(dc) W"
{ echo; echo "## B. DC total (SMC PZ0G) vs host RAPL cap with the 16 clients running; stop at 1180 W"; echo "| host cap W | DC W (max over 40 s) | die W mean |"; echo "|---:|---:|---:|"; } >> $OUT
python3 $B/server-bench.py http://127.0.0.1:8089 $TAG-bg --conc 16 --gen 256 --prompt-tokens 1300 --min-req 400 > /dev/null 2>&1 & CL=$!
for cap in 150 175 200 215; do last=$(dc); if [ "${last:-0}" -ge 1150 ]; then log "not raising to $cap: DC $last"; break; fi
  setrapl $cap; mx=0; for t in $(seq 1 8); do sleep 5; v=$(dc); [ "${v:-0}" -gt $mx ] && mx=$v; if [ "${v:-0}" -ge 1180 ]; then log "STOP at cap $cap: DC $v"; break; fi; done
  echo "| $cap | $mx | $(tail -8 $B/$TAG-clocks.txt | grep -oE '[0-9]+W' | tr -d W | awk '{s+=$1;n++} END{if(n) printf "%.0f", s/n}') |" >> $OUT; log "cap $cap: DC max $mx"
  [ "$mx" -ge 1180 ] && break
done
setrapl 150; stopload; kill $CL 2>/dev/null; wait $CL 2>/dev/null; stop_server $S; kill $SMCPID 2>/dev/null
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
