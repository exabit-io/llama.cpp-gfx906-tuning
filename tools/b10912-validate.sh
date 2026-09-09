#!/bin/bash
# Rebase validation (M5 / TODO 15): our series ported onto the fork's b10912 (/opt/llama.cpp-b10912-gfx906) against the current production
# (/opt/llama.cpp-prod = fork b10254 + series), the fork's pristine b10912 and upstream's pristine b10859.
#  A. test-backend-ops (MUL_MAT / MUL_MAT_ID, all ops) on the new build   B. perplexity 16K   C. greedy + KL vs production
#  D. llama-bench pp2048/tg128 tp4 + rocm0 for the four builds (-r 3)      E. batched-bench tp4 -npl 1..17,24,32 and rocm0 1,2,4,8 (boundary widths) on the new build and production
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-b10912-validate
NEW=/opt/llama.cpp-b10912-gfx906; PR=/opt/llama.cpp-prod; FK=/opt/llama.cpp-b10912; UP=/opt/llama.cpp-b10859
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
AR="GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481"
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md; F=/root/models/wikitext-2-raw/wiki.test.raw
PROMPT="The four dies of a Radeon Pro Vega II Duo share one XGMI ring. In a tensor split, each decode token"
echo "# $TAG  $(date -Is)  fork b10912 + our series vs production (fork b10254 + series), fork pristine b10912, upstream pristine b10859" > $OUT
# A. test-backend-ops
TBO=/root/mx-llama.cpp-b10912/build/bin/test-backend-ops
if [ -x $TBO ]; then log "test-backend-ops MUL_MAT"; LD_LIBRARY_PATH=/root/mx-llama.cpp-b10912/build/bin:$NEW/lib timeout 1800 $TBO test -b ROCm0 -o MUL_MAT > $B/$TAG-tbo-mulmat.log 2>&1; r1=$(grep -E 'tests passed|tests failed|FAIL' $B/$TAG-tbo-mulmat.log | tail -2 | tr '\n' ' ')
  log "test-backend-ops MUL_MAT_ID"; LD_LIBRARY_PATH=/root/mx-llama.cpp-b10912/build/bin:$NEW/lib timeout 1800 $TBO test -b ROCm0 -o MUL_MAT_ID > $B/$TAG-tbo-mmid.log 2>&1; r2=$(grep -E 'tests passed|tests failed|FAIL' $B/$TAG-tbo-mmid.log | tail -2 | tr '\n' ' ')
  log "test-backend-ops all (RMS_NORM, GATED_DELTA_NET, L2_NORM, ADD)"; LD_LIBRARY_PATH=/root/mx-llama.cpp-b10912/build/bin:$NEW/lib timeout 2400 $TBO test -b ROCm0 -o RMS_NORM > $B/$TAG-tbo-norm.log 2>&1; r3=$(grep -E 'tests passed|tests failed' $B/$TAG-tbo-norm.log | tail -1)
  LD_LIBRARY_PATH=/root/mx-llama.cpp-b10912/build/bin:$NEW/lib timeout 2400 $TBO test -b ROCm0 -o GATED_DELTA_NET > $B/$TAG-tbo-gdn.log 2>&1; r4=$(grep -E 'tests passed|tests failed' $B/$TAG-tbo-gdn.log | tail -1)
  { echo; echo "## A. test-backend-ops (ROCm0): MUL_MAT: ${r1:-?}; MUL_MAT_ID: ${r2:-?}; RMS_NORM: ${r3:-?}; GATED_DELTA_NET: ${r4:-?}"; } >> $OUT
else { echo; echo "## A. test-backend-ops not built"; } >> $OUT; fi
# B/C numerics
env $AR LD_LIBRARY_PATH=$NEW/lib $NEW/bin/llama-perplexity -m $M $D4 -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F > $B/$TAG-ppl.log 2>&1
fe=$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-ppl.log | tail -1); { echo; echo "## B. perplexity 16K, new build: ${fe:-none} (production 5.5969)"; } >> $OUT; log "ppl ${fe:-none}"
env $AR LD_LIBRARY_PATH=$PR/lib $PR/bin/llama-completion -m $M $D4 -fa on -b 2048 -ub 2048 -c 4096 -n 200 --temp 0 --seed 1 -no-cnv -p "$PROMPT" > $B/$TAG-prod-greedy.txt 2>/dev/null
env $AR LD_LIBRARY_PATH=$NEW/lib $NEW/bin/llama-completion -m $M $D4 -fa on -b 2048 -ub 2048 -c 4096 -n 200 --temp 0 --seed 1 -no-cnv -p "$PROMPT" > $B/$TAG-new-greedy.txt 2>/dev/null
pos=$(cmp $B/$TAG-new-greedy.txt $B/$TAG-prod-greedy.txt 2>/dev/null | grep -oE 'byte [0-9]+' | grep -oE '[0-9]+'); [ -z "$pos" ] && pos=-1
env $AR LD_LIBRARY_PATH=$PR/lib $PR/bin/llama-perplexity -m $M $D4 -fa on -b 64 -ub 64 -c 2048 --chunks 8 -f $F --kl-divergence-base $B/$TAG-base.kld > $B/$TAG-kld-base.log 2>&1
env $AR LD_LIBRARY_PATH=$NEW/lib $NEW/bin/llama-perplexity -m $M $D4 -fa on -b 64 -ub 64 -c 2048 --chunks 8 -f $F --kl-divergence-base $B/$TAG-base.kld --kl-divergence > $B/$TAG-kld-new.log 2>&1
kl=$(grep -E 'Mean\s+KLD|90.0%|Same top p|Mean.*ln\(PPL' $B/$TAG-kld-new.log | tr -s ' ' | tr '\n' ';' | cut -c1-260)
{ echo; echo "## C. greedy vs production differs at char ${pos} (-1 identical); KL at 64-token batches: ${kl:-none}"; } >> $OUT; log "greedy $pos; kl ${kl:-none}"
# D. llama-bench four builds
{ echo; echo "## D. llama-bench -p 2048 -n 128 -r 3 (tp4 + rocm0); custom AR env on the two builds that carry the gate"; echo "| build | tp4 pp2048 | tp4 tg128 | rocm0 pp2048 | rocm0 tg128 |"; echo "|---|---:|---:|---:|---:|"; } >> $OUT
lb() { local P=$1 name=$2; shift 2; env "$@" LD_LIBRARY_PATH=$P/lib $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3,rocm0 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 128 -d 0 -r 3 -o md > $B/$TAG-$name-lb.md 2>$B/$TAG-$name-lb.err
  local cells; cells=$(grep -E '^\| qwen' $B/$TAG-$name-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$12); printf "%s | ", $12}'); echo "| $name | $cells" >> $OUT; log "lb $name: $cells"; }
lb $PR prod-b10254 $AR; lb $NEW new-b10912 $AR; lb $FK fork-b10912-pristine X=1; lb $UP upstream-b10859-pristine X=1; lb $NEW new-b10912-noar X=1; lb $PR prod-b10254-repeat $AR; lb $NEW new-b10912-repeat $AR
# E. boundary widths
{ echo; echo "## E. batched-bench decode tok/s at 2K, tp4 -npl 1..17,24,32 (MMVQ 1-16, MMQ 17+) and rocm0 -npl 1,2,4,8 at 512"; } >> $OUT
for P in $NEW $PR; do n=$(basename $P); { echo "### $n tp4"; env $AR LD_LIBRARY_PATH=$P/lib $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,24,32 2>/dev/null | grep -E '^\| *2048 ' | awk -F'|' '{gsub(/ /,"",$4); gsub(/ /,"",$9); printf "%s:%s ", $4, $9}'; echo; echo "### $n rocm0"; env $AR LD_LIBRARY_PATH=$P/lib $P/bin/llama-batched-bench -m $M --device rocm0 -fa on -b 2048 -ub 2048 -c 5120 -npp 512 -ntg 128 -npl 1,2,4,8 2>/dev/null | grep -E '^\| *512 ' | awk -F'|' '{gsub(/ /,"",$4); gsub(/ /,"",$9); printf "%s:%s ", $4, $9}'; echo; } >> $OUT; log "boundary $n done"; done
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
