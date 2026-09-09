#!/bin/bash
# TODO 12: tile-table ablation on upstream b10288: stock, +tile-load threads-per-row (a), +MMQ config Q8_0 8 warps/wide tiles (b), +q8_0 k-unroll (c);
# reference: the fork b10254 whole build (/opt/llama.cpp-mxxm). llama-bench pp2048 one die + tp4 (-r 3) at Q8_0, Q6_K, Q4_K_M; batched tp4 24/32 slots at Q8_0.
set -u
B=/root/rocm-tests/bench; TAG=qwen38-27b-ablation
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md
{ echo "# $TAG  $(date -Is)  tile-table ablation on upstream b10288"; echo "| build | model | rocm0 pp2048 | rocm0 tg128 | tp4 pp2048 | tp4 tg128 |"; echo "|---|---|---:|---:|---:|---:|"; } > $OUT
lb() { local P=$1 name=$2 M=$3 q=$4; LD_LIBRARY_PATH=$P/lib $P/bin/llama-bench -m $M --device rocm0,rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 128 -d 0 -r 3 -o md > $B/$TAG-$name-$q-lb.md 2>$B/$TAG-$name-$q-lb.err
  local cells; cells=$(grep -E '^\| qwen' $B/$TAG-$name-$q-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$12); printf "%s | ", $12}'); echo "| $name | $q | $cells" >> $OUT; log "$name $q: $cells"; }
for mq in "Q8_0 /root/models/Qwen3.8-27B-Q8_0.gguf" "Q6_K /root/models/Qwen3.8-27B-UD-Q6_K.gguf" "Q4_K_M /root/models/Qwen3.8-27B-UD-Q4_K_M.gguf"; do set -- $mq; q=$1; M=$2; [ -f $M ] || { log "missing $M"; continue; }
  lb /opt/llama.cpp stock $M $q; lb /opt/llama.cpp-ablation-a a-tileload $M $q; lb /opt/llama.cpp-ablation-b b-config $M $q; lb /opt/llama.cpp-ablation-c c-kunroll $M $q; lb /opt/llama.cpp-mxxm fork-b10254 $M $q
done
{ echo; echo "## batched-bench tp4 Q8_0 -npl 16,24,32 at 2K (decode tok/s)"; echo "| build | 16 | 24 | 32 |"; echo "|---|---:|---:|---:|"; } >> $OUT
M=/root/models/Qwen3.8-27B-Q8_0.gguf
for pn in "/opt/llama.cpp stock" "/opt/llama.cpp-ablation-c c-kunroll" "/opt/llama.cpp-mxxm fork-b10254"; do set -- $pn; LD_LIBRARY_PATH=$1/lib $1/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 16,24,32 > $B/$TAG-$2-bb.md 2>/dev/null
  echo "| $2 | $(for n in 16 24 32; do grep -E '^\| *2048 ' $B/$TAG-$2-bb.md | awk -F'|' -v n=$n '$4+0==n {gsub(/ /,"",$9); printf "%s", $9}'; printf ' | '; done)" >> $OUT; log "bb $2 done"; done
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
