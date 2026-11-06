#!/bin/bash
# dsv41-test.sh BUILD OUTDIR [full] -- DeepSeek-V4.1-Flash-1017 UD-IQ1_S (upstream GGUF format) on the four dies.
# Offload is left to the tree's auto-fit (experts and the Engram table go to the host as needed).
#  1. -sm layer greedy 64 tokens  2. -sm layer perplexity 2K x 4 chunks
#  full only: 3. -sm tensor greedy 64 tokens (fork's head-split routing)  4. llama-server + DSpark sidecar, greedy 192
set -u
B=$1; O=$2; FULL=${3:-}
R=/root/rocm-tests/bench
D=/root/models/DeepSeek-V4.1-Flash-1017-UD-IQ1_S
M=$D/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf; DS=$D/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
F=/root/models/wikitext-2-raw/wiki.test.raw
mkdir -p $O; P=$O/progress; DONE=$O/.done; rm -f $DONE; echo $$ > $O/pid
LDP=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib
D4="--device rocm0,rocm1,rocm2,rocm3"
log(){ echo "$(date -Is) [dsv41] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
. $R/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
export AMD_COMGR_CACHE_DIR=/root/.cache/comgr-100 MIOPEN_CUSTOM_CACHE_DIR=/root/.cache/miopen-100
SPID=""; BPID=""
cleanup(){ trap - INT TERM EXIT; for p in "${BPID:-}" "${SPID:-}" "${SAMP:-}" "${WDOG:-}" "${SMCL:-}"; do kpid "$p"; done
  restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $O/clocks.txt
log "build $B; $(state_line)"
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$O/smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $O/watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $O/smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
run(){ local name=$1; shift; "$@" > $O/$name.out 2> $O/$name.log < /dev/null & BPID=$!; wait $BPID; local rc=$?; BPID=""; return $rc; }
place(){ grep -hiE 'fit|offloaded|model buffer size|n_cpu_moe|overrid|projected' $1 | head -8 | sed 's/^[0-9.]* //' | tr '\n' ';' | cut -c1-400; }
PROMPT="The transistor was invented in 1947 at Bell Labs. In the decades that followed,"   # raw continuation: the bare question made the model stop at once

log "1. -sm layer greedy 64 tokens"
LD_LIBRARY_PATH=$LDP run layer timeout 3600 $B/bin/llama-completion -m $M $D4 -sm layer -fa on -c 4096 -n 64 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup -p "$PROMPT"
log "  rc=$? $(wc -c < $O/layer.out) bytes, md5 $(md5sum < $O/layer.out | cut -c1-8): $(tr '\n' ' ' < $O/layer.out | cut -c1-220)"
log "  speed: $(grep -E 'prompt eval time|       eval time' $O/layer.log | tail -2 | sed 's/.*= *//' | tr '\n' ' ' | cut -c1-200)"
log "  placement: $(place $O/layer.log)"

log "2. -sm layer perplexity 2K x 4 chunks"
LD_LIBRARY_PATH=$LDP run ppl timeout 7200 $B/bin/llama-perplexity -m $M $D4 -sm layer -fa on -c 2048 -b 2048 -ub 512 --chunks 4 -f $F
log "  rc=$? $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $O/ppl.log | tail -1) $(grep -E 'seconds per pass' $O/ppl.log | head -1 | sed 's/.*: //')"

if [ "$FULL" = full ]; then
  log "3. -sm tensor greedy 64 tokens"
  LD_LIBRARY_PATH=$LDP run tensor timeout 3600 $B/bin/llama-completion -m $M $D4 -sm tensor -fa on -c 4096 -n 64 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup -p "$PROMPT"
  log "  rc=$? $(wc -c < $O/tensor.out) bytes, md5 $(md5sum < $O/tensor.out | cut -c1-8): $(tr '\n' ' ' < $O/tensor.out | cut -c1-220)"
  log "  speed: $(grep -E 'prompt eval time|       eval time' $O/tensor.log | tail -2 | sed 's/.*= *//' | tr '\n' ' ' | cut -c1-200); errors: $(grep -iE 'error|abort|not supported' $O/tensor.log | head -2 | tr '\n' ' ' | cut -c1-200)"

  log "4. llama-server -sm layer with the DSpark sidecar, greedy 192 tokens"
  for mode in base dspark; do
    extra=""; [ $mode = dspark ] && extra="--spec-type draft-dspark -md $DS"
    LD_LIBRARY_PATH=$LDP $B/bin/llama-server -m $M $D4 -sm layer -fa on -c 8192 -np 1 --port 8096 $extra > $O/srv-$mode.out 2> $O/srv-$mode.log < /dev/null &
    SPID=$!
    if wait_server 8096 $SPID; then
      curl -s --max-time 1800 http://127.0.0.1:8096/completion -H 'Content-Type: application/json' \
        -d '{"prompt":"The transistor was invented in 1947 at Bell Labs. In the decades that followed,","n_predict":192,"temperature":0,"top_k":1,"seed":0}' > $O/srv-$mode.json
      log "  $mode: $(python3 -I -c 'import json,sys,hashlib; d=json.load(open(sys.argv[1])); t=d.get("timings",{}); print("tokens", t.get("predicted_n"), "tok/s %.2f"%t.get("predicted_per_second",0), "draft", t.get("draft_n_accepted"), "/", t.get("draft_n"), "| md5", hashlib.md5(d.get("content","").encode()).hexdigest()[:8], "|", d.get("content","")[:100].replace(chr(10)," "))' $O/srv-$mode.json 2>&1)"
    else
      log "  $mode: server did not come up: $(tail -3 $O/srv-$mode.log | tr '\n' ' ' | cut -c1-300)"
    fi
    kpid "$SPID"; wait "$SPID" 2>/dev/null; SPID=""; sleep 5
  done
fi
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore; touch $DONE; log "=== DSV41 DONE ==="
