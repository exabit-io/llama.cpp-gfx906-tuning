#!/bin/bash
# NEXT-STEPS S2: size gate for the fork's peer-write custom allreduce (GGML_TP_AR_MAX_NE, fusion build /opt/llama.cpp-mxxm-fh-nq).
# The knob sweep (queue-13) showed +14% single stream and -8% at 8/16 slots with the default gate; find the crossover in decode rows.
# batched-bench tp4 -npl 1,2,4,8,16 at 2K per variant; llama-bench tg128 for the baseline and the 1-row gate.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-nq-argate; P=/opt/llama.cpp-mxxm-fh-nq
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib
NE=$(python3 $B/gguf-kv.py $M embedding_length | awk '{print $2}')
log "n_embd $NE (one decode row per allreduce message)"
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md
{ echo "# $TAG  $(date -Is)  fusion build tp4; custom AR gate in rows of n_embd=$NE (libggml-hip: $(ldd $P/bin/llama-batched-bench | grep -o '/opt/[^ ]*libggml-hip[^ ]*'))"
  echo "| variant | env | 1 slot | 2 | 4 | 8 | 16 | tg128 (llama-bench) |"; echo "|---|---|---:|---:|---:|---:|---:|---:|"; } > $OUT
PW="GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1"
variant() { local name=$1 lb=$2; shift 2
  env "$@" $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 34816 -npp 2048 -ntg 128 -npl 1,2,4,8,16 > $B/$TAG-$name-bb.md 2>$B/$TAG-$name-bb.err
  local cells; cells=$(for n in 1 2 4 8 16; do grep -E '^\| *2048 ' $B/$TAG-$name-bb.md | awk -F'|' -v n=$n '$4+0==n {gsub(/ /,"",$9); printf "%s", $9}'; printf ' | '; done)
  local tg=""; if [ "$lb" = 1 ]; then env "$@" $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 0 -n 128 -r 3 -o md > $B/$TAG-$name-lb.md 2>$B/$TAG-$name-lb.err; tg=$(grep -E '^\| qwen.*tg128' $B/$TAG-$name-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$12); print $12}'); fi
  echo "| $name | $* | $cells ${tg:--} |" >> $OUT; log "$name: $cells tg $tg"; }
variant base      1 BENCH_VARIANT=base
variant pw-default 0 $PW
variant pw-1row   1 $PW GGML_TP_AR_MAX_NE=$((NE+1))
variant pw-2rows  0 $PW GGML_TP_AR_MAX_NE=$((2*NE+1))
variant pw-4rows  0 $PW GGML_TP_AR_MAX_NE=$((4*NE+1))
variant pw-8rows  0 $PW GGML_TP_AR_MAX_NE=$((8*NE+1))
variant base-repeat 0 BENCH_VARIANT=base2
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
