#!/bin/bash
# TODO 14: paired-run spread for the production table cells. Builds interleaved per round: stock, 2026-09-07 production, current production (+gfx906.env).
# llama-bench -r 5 (tp4 pp2048/tg128 + rocm0), 3 rounds; batched-bench tp4 -npl 8,12,16 and rocm0 -npl 4,8, 2 rounds.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-ci-paired
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
AR="GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481"
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md
{ echo "# $TAG  $(date -Is)  paired rounds: stock /opt/llama.cpp, prod0907 /opt/llama.cpp-mxxm-fh, prod /opt/llama.cpp-prod (+custom AR env)"; echo; echo "## llama-bench -p 2048 -n 128 -r 5, three interleaved rounds"; echo "| round | build | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |"; echo "|---|---|---:|---:|---:|---:|"; } > $OUT
lb() { local P=$1 name=$2 r=$3; shift 3; env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3,rocm0 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 128 -d 0 -r 5 -o md > $B/$TAG-$name-r$r-lb.md 2>$B/$TAG-$name-r$r-lb.err
  local cells; cells=$(grep -E '^\| qwen' $B/$TAG-$name-r$r-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$12); printf "%s | ", $12}'); echo "| $r | $name | $cells" >> $OUT; log "lb r$r $name: $cells"; }
for r in 1 2 3; do lb /opt/llama.cpp stock $r X=1; lb /opt/llama.cpp-mxxm-fh prod0907 $r X=1; lb /opt/llama.cpp-prod prod $r $AR; done
{ echo; echo "## batched-bench decode tok/s, two rounds: tp4 -npl 8,12,16 at 2K; rocm0 -npl 4,8 at 512"; echo "| round | build | tp4 8 | tp4 12 | tp4 16 | rocm0 4 | rocm0 8 |"; echo "|---|---|---:|---:|---:|---:|---:|"; } >> $OUT
bb() { local P=$1 name=$2 r=$3; shift 3
  env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 34816 -npp 2048 -ntg 128 -npl 8,12,16 > $B/$TAG-$name-r$r-bb4.md 2>/dev/null
  env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-batched-bench -m $M --device rocm0 -fa on -b 2048 -ub 2048 -c 5120 -npp 512 -ntg 128 -npl 4,8 > $B/$TAG-$name-r$r-bb1.md 2>/dev/null
  local c4 c1; c4=$(for n in 8 12 16; do grep -E '^\| *2048 ' $B/$TAG-$name-r$r-bb4.md | awk -F'|' -v n=$n '$4+0==n {gsub(/ /,"",$9); printf "%s", $9}'; printf ' | '; done); c1=$(for n in 4 8; do grep -E '^\| *512 ' $B/$TAG-$name-r$r-bb1.md | awk -F'|' -v n=$n '$4+0==n {gsub(/ /,"",$9); printf "%s", $9}'; printf ' | '; done)
  echo "| $r | $name | $c4 $c1" >> $OUT; log "bb r$r $name: $c4 $c1"; }
for r in 1 2; do bb /opt/llama.cpp stock $r X=1; bb /opt/llama.cpp-mxxm-fh prod0907 $r X=1; bb /opt/llama.cpp-prod prod $r $AR; done
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
