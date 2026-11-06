#!/bin/bash
# nemotron-test.sh BUILD OUTDIR -- NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB (nemotron_h_moe) on one die:
#  1. greedy 96 tokens  2. perplexity 4K x 4 chunks  3. llama-server without and with the LiLiCorr drafter
#  (--spec-type draft-dflash, NVIDIA's recipe), greedy, 256 tokens.  4. -sm tensor greedy on four dies (smoke).
set -u
B=$1; O=$2
R=/root/rocm-tests/bench
N=/root/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
M=$N/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf; L=$N/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf
F=/root/models/wikitext-2-raw/wiki.test.raw
mkdir -p $O; P=$O/progress; DONE=$O/.done; rm -f $DONE; echo $$ > $O/pid
LDP=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib
log(){ echo "$(date -Is) [nemotron] $*" | tee -a $P; }
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
PROMPT="Explain in two sentences why the sky is blue."

log "1. greedy 96 tokens, one die"
LD_LIBRARY_PATH=$LDP run greedy timeout 900 $B/bin/llama-completion -m $M --device rocm0 -ngl all -fa on -c 8192 -n 96 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup -p "$PROMPT"
log "  rc=$? $(wc -c < $O/greedy.out) bytes, md5 $(md5sum < $O/greedy.out | cut -c1-8): $(tr '\n' ' ' < $O/greedy.out | cut -c1-200)"
log "  $(grep -E 'eval time' $O/greedy.log | tail -2 | sed 's/.*://' | tr '\n' ' ' | cut -c1-200)"

log "2. perplexity 4K x 4 chunks, one die"
LD_LIBRARY_PATH=$LDP run ppl timeout 3600 $B/bin/llama-perplexity -m $M --device rocm0 -ngl all -fa on -c 4096 -b 2048 -ub 512 --chunks 4 -f $F
log "  rc=$? $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $O/ppl.log | tail -1)"

srv(){ # srv NAME PORT ARGS... -- start a server on rocm0, one greedy request, report
  local name=$1 port=$2; shift 2
  LD_LIBRARY_PATH=$LDP $B/bin/llama-server -m $M --device rocm0 -ngl all -fa on -c 32768 -b 2048 -ub 512 -np 1 --port $port "$@" > $O/$name-server.out 2> $O/$name-server.log < /dev/null &
  SPID=$!
  if wait_server $port $SPID; then
    curl -s --max-time 900 http://127.0.0.1:$port/completion -H 'Content-Type: application/json' \
      -d '{"prompt":"Write a short paragraph about the history of the transistor.","n_predict":256,"temperature":0,"top_k":1,"seed":0}' > $O/$name.json
    log "  $name: $(python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); t=d.get("timings",{}); print("tokens", t.get("predicted_n"), "tok/s %.2f"%t.get("predicted_per_second",0), "prompt tok/s %.1f"%t.get("prompt_per_second",0), "draft", t.get("draft_n_accepted"), "/", t.get("draft_n"), "| md5", __import__("hashlib").md5(d.get("content","").encode()).hexdigest()[:8], "|", d.get("content","")[:100].replace(chr(10)," "))' $O/$name.json 2>&1)"
    log "  $name path: $(grep -iE 'lilicorr|dflash|eager|deferred|draft' $O/$name-server.log | grep -viE 'print_timing|statistics' | head -3 | sed 's/^[0-9.]* //' | tr '\n' ';' | cut -c1-260)"
  else
    log "  $name: server did not come up: $(tail -3 $O/$name-server.log | tr '\n' ' ' | cut -c1-300)"
  fi
  kpid "$SPID"; wait "$SPID" 2>/dev/null; SPID=""; sleep 3
}
log "3. llama-server, one die, greedy 256 tokens: baseline, then the LiLiCorr drafter"
srv base 8094
srv lilicorr 8095 --spec-type draft-dflash --spec-draft-n-max 6 --spec-draft-model $L --spec-draft-ngl 999 --spec-draft-device rocm0

log "4. -sm tensor over four dies, greedy 96 tokens (smoke)"
LD_LIBRARY_PATH=$LDP run tensor timeout 900 $B/bin/llama-completion -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -c 8192 -n 96 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup -p "$PROMPT"
log "  rc=$? $(wc -c < $O/tensor.out) bytes, md5 $(md5sum < $O/tensor.out | cut -c1-8): $(tr '\n' ' ' < $O/tensor.out | cut -c1-160) $(grep -iE 'not supported|tensor split|abort|error' $O/tensor.log | head -2 | tr '\n' ' ' | cut -c1-200)"

trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore; touch $DONE; log "=== NEMOTRON DONE ==="
