#!/bin/bash
# q6k-campaign.sh -- Q6_K vs Q8_0, Qwen3.8-27B abliterated, on the published build (mx-llama-cpp-gfx906, /opt/mx-llama.cpp).
# Lead-approved 2026-10-22: full chain, REPORT ONLY (no verdict). Every cell at >= 64K, f16 KV, 125 W/die (macpro-fand pin),
# RCCL + gated custom AR (the binrun environment). Phases:
#   Q  quality vs the BF16 original at 64K context, 4 chunks (KLD, top-token agreement, PPL); Lam12 Q6_K vs Lam12 Q8_0
#   S  speed: llama-batched-bench, multi 4 x 65536 (n=5) and single 1 x 260864 (n=6), -ntg 1024, seeded random order
#   M  MTP off vs draft-mtp n-max 3: mtp-depth-client, 4 x 64K and 1 x 252K, 1024 generated, two waves, n=3, random order
#   C  capacity: largest context per slot that loads at 4 slots, f16 KV, per quant (bisection on llama-server load)
# PHASES (env) selects phases, default "Q S M C".
set -u
W=/root/rebase-v070/q6k; mkdir -p $W
R=/root/rocm-tests/bench; TL=/root/llama.cpp-benchmarking/tools; CM=$TL/cell-metrics.py; NF=$TL/assert-no-fa-fallback.sh
B=/opt/mx-llama.cpp; MD=/root/models; F=$MD/wikitext-2-raw/wiki.test.raw
BF=$MD/Huihui-Qwen3.8-27B-abliterated-BF16.gguf
declare -A MODEL=( [q8]=$MD/Huihui-Qwen3.8-27B-abliterated-Q8_0.gguf [q6]=$MD/Huihui-Qwen3.8-27B-abliterated-Q6_K.gguf
                   [lam12-q8]=$MD/Qwen3.8-27B-abliterated-mtp-Q8_0.gguf [lam12-q6]=$MD/Qwen3.8-27B-abliterated-mtp-Q6_K.gguf )
D4="--device rocm0,rocm1,rocm2,rocm3"
PHASES=${PHASES:-"Q S M C"}; NMULTI=${NMULTI:-5}; NSINGLE=${NSINGLE:-6}; NMTP=${NMTP:-3}
P=$W/progress; DONE=$W/.done; rm -f $DONE; echo $$ > $W/pid
log(){ echo "$(date -Is) [q6k] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml GGML_CUDA_ALLREDUCE=nccl
export GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
BPID=""; SPID=""
cleanup(){ trap - INT TERM EXIT; for p in "${BPID:-}" "${SPID:-}" "${SAMP:-}" "${WDOG:-}" "${SMCL:-}"; do kpid "$p"; done
  rm -f $W/*.kld; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $W/watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
log "start: build $B ($(cut -c1-9 /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT)), ROCm $(dpkg-query -W -f='${Version}' amdrocm-core10.0-gfx906), phases '$PHASES'; $(state_line)"
run(){ local name=$1; shift; "$@" > $W/$name.out 2> $W/$name.log < /dev/null & BPID=$!; wait $BPID; local rc=$?; BPID=""; return $rc; }
order(){ python3 -I -c 'import random,sys; a=sys.argv[2:]; random.Random(int(sys.argv[1])).shuffle(a); print(" ".join(a))' "$@"; }

# ---------------- Q: quality at 64K ----------------
if [[ " $PHASES " == *" Q "* ]]; then
  log "=== Q: quality, 64K context x 4 chunks"
  PPLARGS="$D4 -sm tensor -ngl all -fa on -c 65536 -b 2048 -ub 2048 --chunks 4 -f $F"
  kl(){ # kl NAME MODEL BASEFILE
    run $1 timeout 7200 $B/bin/llama-perplexity -m $2 $PPLARGS --kl-divergence-base $3 --kl-divergence
    log "  $1 rc=$?: $(grep -E 'Mean PPL\(Q\)|Mean PPL\(base\)|Mean +KLD|99.0% +KLD|99.9% +KLD|Maximum KLD|Same top p|RMS Δp|Mean +Δp' $W/$1.out | tr -s ' ' | tr '\n' ';')"
  }
  if [ -f $BF ]; then
    run q-base-bf16 timeout 7200 $B/bin/llama-perplexity -m $BF $PPLARGS --kl-divergence-base $W/huihui-bf16.kld
    log "  BF16 base rc=$?: $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $W/q-base-bf16.log | tail -1)"
    kl q-kld-q8 ${MODEL[q8]} $W/huihui-bf16.kld
    kl q-kld-q6 ${MODEL[q6]} $W/huihui-bf16.kld
    rm -f $W/huihui-bf16.kld
  else
    log "  NO BF16 GGUF at $BF -- Huihui quality vs BF16 skipped"
  fi
  run q-base-lam12-q8 timeout 7200 $B/bin/llama-perplexity -m ${MODEL[lam12-q8]} $PPLARGS --kl-divergence-base $W/lam12-q8.kld
  log "  Lam12 Q8_0 base rc=$?: $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $W/q-base-lam12-q8.log | tail -1)"
  kl q-kld-lam12-q6 ${MODEL[lam12-q6]} $W/lam12-q8.kld
  rm -f $W/lam12-q8.kld
fi

# ---------------- S: speed, batched-bench ----------------
if [[ " $PHASES " == *" S "* ]]; then
  TSV=$W/speed.tsv
  cell(){ # cell AXIS ARM BLOCK
    local axis=$1 arm=$2 blk=$3 s=4 d=65536
    [ "$axis" = single ] && { s=1; d=260864; }
    local out=$W/s-$axis-$arm-$blk.md
    run s-$axis-$arm-$blk timeout 7200 $B/bin/llama-batched-bench -m ${MODEL[$arm]} $D4 -sm tensor -ngl all \
      -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s
    if ! $NF $W/s-$axis-$arm-$blk.log >/dev/null 2>&1; then log "  FA FALLBACK $axis $arm b$blk -- discarded"; printf "%s\t%s\t%s\tFALLBACK\tFALLBACK\n" $axis $arm $blk >> $TSV; return 0; fi
    local m
    if ! m=$(python3 -I $CM $W/s-$axis-$arm-$blk.out $d $s 2>>$P); then log "  CELL FAILED $axis $arm b$blk"; printf "%s\t%s\t%s\tFAILED\tFAILED\n" $axis $arm $blk >> $TSV; return 0; fi
    printf "%s\t%s\t%s\t%s\n" $axis $arm $blk "$m" >> $TSV
    log "  $axis $arm b$blk: $(echo "$m" | awk -F'\t' '{printf "decode %.4f tok/s/seq, prefill %.1f t/s",$1,$2}')"
  }
  for axis in multi single; do
    nb=$NMULTI; [ $axis = single ] && nb=$NSINGLE
    log "=== S: $axis, 2 arms x $nb blocks"
    for blk in $(seq 1 $nb); do
      o=$(order $(( 20261022 + blk + ${#axis}*10 )) q8 q6); log "  block $blk order: $o"
      for a in $o; do cell $axis $a $blk; done
    done
  done
fi

# ---------------- M: MTP off vs on, server ----------------
if [[ " $PHASES " == *" M "* ]]; then
  mcell(){ # mcell PROFILE ARM MODE BLOCK
    local prof=$1 arm=$2 mode=$3 blk=$4 np=4 depth=65536 extra=""
    [ $prof = single ] && { np=1; depth=258048; }
    [ $mode = mtp3 ] && extra="--spec-type draft-mtp --spec-draft-n-max 3"
    local tag=q6k-m-$prof-$arm-$mode-$blk port=$((8100 + blk))
    $B/bin/llama-server -m ${MODEL[$arm]} $D4 -sm tensor -ngl all -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -cb \
      --host 127.0.0.1 --port $port -np $np -c $(( np*(depth+4096) )) $extra > $W/$tag.server.log 2>&1 < /dev/null &
    SPID=$!
    if wait_server $port $SPID; then
      python3 -I $R/mtp-depth-client.py http://127.0.0.1:$port $tag $W/$tag.server.log --context-tokens $depth --conc $np --gen 1024 --waves 2 \
        > $W/$tag.md 2> $W/$tag.err < /dev/null & BPID=$!; wait $BPID; local rc=$?; BPID=""
      [ -f $R/$tag.json ] && mv $R/$tag.json $W/
      log "  $prof $arm $mode b$blk rc=$rc: $(tr '\n' ' ' < $W/$tag.md | cut -c1-260)"
    else
      log "  $prof $arm $mode b$blk: server failed: $(grep -ihE 'error|assert|out of memory' $W/$tag.server.log | head -1 | cut -c1-160)"
    fi
    kpid "$SPID"; wait "$SPID" 2>/dev/null; SPID=""; sleep 4
  }
  for prof in multi single; do
    log "=== M: $prof, 4 arms x $NMTP blocks"
    for blk in $(seq 1 $NMTP); do
      o=$(order $(( 20261122 + blk + ${#prof}*10 )) q8:off q8:mtp3 q6:off q6:mtp3); log "  block $blk order: $o"
      for am in $o; do mcell $prof ${am%%:*} ${am##*:} $blk; done
    done
  done
fi

# ---------------- C: capacity at 4 slots ----------------
if [[ " $PHASES " == *" C "* ]]; then
  log "=== C: largest context per slot that loads, 4 slots, f16 KV (bisection, 4096 steps)"
  fits(){ # fits ARM CTX_PER_SLOT -> 0 if llama-server loads and answers
    local arm=$1 c=$2 port=8120
    $B/bin/llama-server -m ${MODEL[$arm]} $D4 -sm tensor -ngl all -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -cb \
      --host 127.0.0.1 --port $port -np 4 -c $(( 4*c )) > $W/c-$arm-$c.log 2>&1 < /dev/null &
    SPID=$!
    local ok=1
    if wait_server $port $SPID && curl -sf --max-time 120 http://127.0.0.1:$port/completion -H 'Content-Type: application/json' -d '{"prompt":"Say OK.","n_predict":4}' > /dev/null; then ok=0; fi
    kpid "$SPID"; wait "$SPID" 2>/dev/null; SPID=""; sleep 3
    return $ok
  }
  for arm in q8 q6; do
    lo=65536; hi=$((262144 + 4096))
    fits $arm $lo || { log "  $arm: does not load at 4 x $lo"; continue; }
    if fits $arm 262144; then lo=262144; hi=262144; fi
    while [ $(( hi - lo )) -gt 4096 ]; do
      mid=$(( (lo + hi) / 2 / 4096 * 4096 )); [ $mid -le $lo ] && mid=$(( lo + 4096 ))
      if fits $arm $mid; then lo=$mid; else hi=$mid; fi
    done
    log "  $arm: 4 slots x $lo tokens loads ($(grep -m4 -E 'KV buffer size|compute buffer size|model buffer size' $W/c-$arm-$lo.log | sed 's/.*: *//' | tr '\n' ';' | cut -c1-200))"
  done
fi

trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; restore; touch $DONE; log "=== CAMPAIGN DONE ==="
