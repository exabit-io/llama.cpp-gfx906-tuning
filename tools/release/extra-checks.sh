#!/bin/bash
# extra-checks.sh BUILD OUTDIR -- runtime checks for the paths the v0.9.0 merge combined (see notes-C):
#  1. MTP with -np 1 and n_probs (speculative tokens must carry probabilities)
#  2. MTP with -np 4, four concurrent requests (parallel contexts: no snapshot ring, cell.rs_max rollback bound)
#  3. test-llama-context (includes rollback_bound with the ring-aware bound)
#  4. llama-gguf-split split + merge round trip, greedy output compared with the original file
set -u
B=$1; O=$2
R=/root/rocm-tests/bench
M=/root/models/Qwen3.8-27B-Q8_0.gguf
S=/root/models/Qwen3.5-0.8B-Q8_0.gguf
mkdir -p $O; P=$O/progress; DONE=$O/.done; rm -f $DONE; echo $$ > $O/pid
LDP=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib
D4="--device rocm0,rocm1,rocm2,rocm3"
log(){ echo "$(date -Is) [extra] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
. $R/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1 LLAMA_PLE_SHARD=1
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
summ(){ python3 -I -c 'import json,sys
d=json.load(open(sys.argv[1])); t=d.get("timings",{}); p=d.get("completion_probabilities") or []
print("tokens", t.get("predicted_n"), "tok/s %.2f"%t.get("predicted_per_second",0), "draft_n", t.get("draft_n"), "accepted", t.get("draft_n_accepted"),
      "| probs entries", len(p), "with top_probs", sum(1 for e in p if e.get("top_probs")), "| text:", d.get("content","")[:90].replace(chr(10)," "))' "$1" 2>&1; }

if [ "${FROM_STEP:-1}" -le 2 ]; then
log "1. MTP -np 1, n_probs 5"
LD_LIBRARY_PATH=$LDP $B/bin/llama-server -m $M $D4 -sm tensor -ngl all -fa on -c 8192 -np 1 --port 8092 \
  --spec-type draft-mtp --spec-draft-n-max 2 > $O/np1-server.out 2> $O/np1-server.log < /dev/null &
SPID=$!
if wait_server 8092 $SPID; then
  curl -s --max-time 600 http://127.0.0.1:8092/completion -H 'Content-Type: application/json' \
    -d '{"prompt":"Write a short paragraph about the history of the transistor.","n_predict":128,"temperature":0,"top_k":1,"seed":0,"n_probs":5}' > $O/np1-probs.json
  log "  $(summ $O/np1-probs.json)"
else
  log "  server did not come up: $(tail -3 $O/np1-server.log | tr '\n' ' ' | cut -c1-300)"
fi
kpid "$SPID"; wait "$SPID" 2>/dev/null; SPID=""; sleep 3

log "2. MTP -np 4, four concurrent requests"
LD_LIBRARY_PATH=$LDP $B/bin/llama-server -m $M $D4 -sm tensor -ngl all -fa on -c 32768 -np 4 --port 8093 \
  --spec-type draft-mtp --spec-draft-n-max 2 > $O/np4-server.out 2> $O/np4-server.log < /dev/null &
SPID=$!
if wait_server 8093 $SPID; then
  i=0; CPIDS=""
  for q in "Write a short paragraph about the history of the transistor." "Explain how a refrigerator works." \
           "List five uses of copper and say why each needs copper." "Describe the water cycle for a ten-year-old."; do
    curl -s --max-time 900 http://127.0.0.1:8093/completion -H 'Content-Type: application/json' \
      -d "{\"prompt\":\"$q\",\"n_predict\":192,\"temperature\":0,\"top_k\":1,\"seed\":0}" > $O/np4-$i.json &
    CPIDS="$CPIDS $!"; i=$((i+1))
  done
  wait $CPIDS 2>/dev/null   # only the four requests: the sampler, watchdog and SMC log are background jobs too
  for i in 0 1 2 3; do log "  req $i: $(summ $O/np4-$i.json)"; done
  log "  refused rollbacks: $(grep -c 'refusing rollback' $O/np4-server.log); checkpoint restores: $(grep -ci 'restor' $O/np4-server.log)"
else
  log "  server did not come up: $(tail -3 $O/np4-server.log | tr '\n' ' ' | cut -c1-300)"
fi
kpid "$SPID"; wait "$SPID" 2>/dev/null; SPID=""; sleep 3

fi
log "3. test-llama-context"
LD_LIBRARY_PATH=$LDP timeout 1800 $B/bin/test-llama-context > $O/llama-context.log 2>&1 < /dev/null &
BPID=$!; wait $BPID; rc=$?; BPID=""
log "  rc=$rc $(grep -ciE 'fail' $O/llama-context.log) fail lines; rollback_bound: $(grep -i 'rollback_bound\|rollback bounds' $O/llama-context.log | tail -2 | tr '\n' ' ' | cut -c1-240)"

log "4. gguf-split round trip (Qwen3.5-0.8B Q8_0)"
rm -rf $O/split; mkdir -p $O/split
LD_LIBRARY_PATH=$LDP $B/bin/llama-gguf-split --split --split-max-tensors 64 $S $O/split/part > $O/split.log 2>&1; r1=$?
LD_LIBRARY_PATH=$LDP $B/bin/llama-gguf-split --merge $O/split/part-00001-of-*.gguf $O/split/merged.gguf >> $O/split.log 2>&1; r2=$?
for f in $S $O/split/merged.gguf; do
  LD_LIBRARY_PATH=$LDP timeout 300 $B/bin/llama-completion -m $f --device rocm0 -ngl all -c 2048 -n 32 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup \
    -p "The capital of France is" > $O/split-$(basename $f).out 2> /dev/null < /dev/null
done
log "  split rc=$r1 parts=$(ls $O/split/part-*.gguf 2>/dev/null | wc -l) merge rc=$r2; greedy output identical: $(cmp -s $O/split-$(basename $S).out $O/split-merged.gguf.out && echo yes || echo NO); size orig/merged $(stat -c %s $S)/$(stat -c %s $O/split/merged.gguf 2>/dev/null)"
rm -rf $O/split

trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore; touch $DONE; log "=== EXTRA DONE ==="
