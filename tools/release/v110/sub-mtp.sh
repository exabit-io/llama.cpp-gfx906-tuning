#!/bin/bash
# sub-mtp.sh TAG N ARM=BUILDDIR ... -- the MTP cell of the substrate comparison (substrate-v110.sh, which holds the
# reading rule): llama-server of each arm's own build, Qwen3.8-27B Q8_0 on four dies (-sm tensor, f16 KV, -np 1),
# --spec-type draft-mtp --spec-draft-n-max 3. Per arm and block: one server, request 1 = the whole prompt (prefill
# rate), then the other questions on the same text (the slot keeps the text, only the question is evaluated); every
# request generates NPRED tokens greedily. N blocks, a fresh random arm order per block.
# Appends to TAG-mtp.tsv: mtp, arm, block, prefill tok/s, mean decode tok/s of the requests, accepted/drafted per
# request, text hashes, decode tok/s per request.
# CHARS (default 200000 characters of wikitext, about 47K tokens), NPRED (512), CTX (65536), SCLK_GUARD (1; 0 only for
# a functional try on a clamped box: then no rate is a number).
set -u
W=/root/rebase-v110; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RT=/opt/rocm/core-10.0
F=/root/models/wikitext-2-raw/wiki.test.raw
TAG=$1; N=$2; shift 2
CHARS=${CHARS:-200000}; NPRED=${NPRED:-512}; CTX=${CTX:-65536}; PORT=8098
P=$W/$TAG-mtp.progress; TSV=$W/$TAG-mtp.tsv; DONE=$W/.$TAG-mtp-done; O=$W/$TAG-mtp; mkdir -p $O
log(){ echo "$(date -Is) [sub-mtp] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; touch $DONE; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; touch $DONE; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is in progress"; touch $DONE; exit 1; }
declare -A BD=(); ARMS=""
for spec in "$@"; do
  a=${spec%%=*}; bd=${spec#*=}
  [ -x $bd/bin/llama-server ] || { log "FATAL: arm $a has no llama-server in $bd"; touch $DONE; exit 1; }
  BD[$a]=$bd; ARMS="$ARMS $a"; log "$a: $bd on $RT"
done
python3 -I - "$F" "$O" "$CHARS" "$NPRED" <<'EOF'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()[:int(sys.argv[3])]
qs = ["Summarize every article in the text above in detail, one paragraph per article, in the order they appear.",
      "Go through the text above article by article and list every person named in it, with one sentence on who they are.",
      "Write a detailed critical review of each article in the text above, one paragraph per article, in the order they appear.",
      "List every date and year mentioned in the text above in the order they appear, each with the event it belongs to.",
      "Explain, article by article, what a reader new to the subject would need to know first to follow the text above."]
for i, q in enumerate(qs, 1):
    json.dump({"messages": [{"role": "user", "content": text + "\n\n" + q}], "max_tokens": int(sys.argv[4]), "temperature": 0, "top_k": 1, "seed": 0,
               "cache_prompt": True}, open("%s/req-%d.json" % (sys.argv[2], i), "w"))
EOF
NREQ=5
log "n=$N, arms:$ARMS; prompt $CHARS characters, $NREQ requests of $NPRED tokens; load $(cut -d' ' -f1-3 /proc/loadavg)"

. $R/gpu-test-env.sh
export GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
SPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${SPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/$TAG-mtp-clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/$TAG-mtp-smc.log SCLK_GUARD=${SCLK_GUARD:-1} nohup $R/clamp-watchdog-v2.sh >> $W/$TAG-mtp-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/$TAG-mtp-smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
log "$(state_line)"
field(){ # field FILE -> prompt_n prompt_tps predicted_n predicted_tps draft_accepted draft_n sha, tab separated; fails on an error reply
  python3 -I -c '
import hashlib, json, sys
d = json.load(open(sys.argv[1])); t = d["timings"]; m = d["choices"][0]["message"]
text = (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")
print("\t".join(str(x) for x in (t["prompt_n"], "%.3f" % t["prompt_per_second"], t["predicted_n"], "%.4f" % t["predicted_per_second"],
      t.get("draft_n_accepted"), t.get("draft_n"), hashlib.sha256(text.encode()).hexdigest()[:12])))' "$1" 2>/dev/null
}
cell(){ # cell ARM BLOCK
  local arm=$1 blk=$2 bd=${BD[$1]} r f pre="" pn="" dec="" acc="" sha="" ok=1 mean
  env LD_LIBRARY_PATH=$bd/bin:$RT/lib AMD_COMGR_CACHE_DIR=/root/.cache/comgr-sub-$arm $bd/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $CTX -np 1 --cache-ram 0 --port $PORT \
    --spec-type draft-mtp --spec-draft-n-max 3 > $O/$arm-$blk-server.out 2> $O/$arm-$blk-server.log < /dev/null &
  SPID=$!
  if ! wait_server $PORT $SPID; then
    log "SERVER DID NOT COME UP: $arm b$blk: $(tail -2 $O/$arm-$blk-server.log | tr '\n' ' ' | cut -c1-200)"
    kpid $SPID; wait $SPID 2>/dev/null; SPID=""; printf "mtp\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$blk" >> $TSV; return 0
  fi
  for r in $(seq 1 $NREQ); do
    curl -s --max-time 1800 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @$O/req-$r.json > $O/$arm-$blk-$r.json
    if ! f=$(field $O/$arm-$blk-$r.json) || [ -z "$f" ]; then ok=0; log "REQUEST $r FAILED: $arm b$blk: $(head -c 200 $O/$arm-$blk-$r.json)"; break; fi
    [ $r = 1 ] && { pre=$(echo "$f" | cut -f2); pn=$(echo "$f" | cut -f1); }
    dec="$dec $(echo "$f" | cut -f4)"; acc="$acc $(echo "$f" | cut -f5)/$(echo "$f" | cut -f6)"; sha="$sha $(echo "$f" | cut -f7)"
    [ "$(echo "$f" | cut -f3)" -ge $(( NPRED / 2 )) ] || { ok=0; log "REQUEST $r FAILED: $arm b$blk generated $(echo "$f" | cut -f3) tokens, fewer than half of $NPRED"; break; }
    [ "$(echo "$f" | cut -f3)" = "$NPRED" ] || log "note: $arm b$blk request $r ended by itself after $(echo "$f" | cut -f3) tokens"
    [ $r != 1 ] && [ "$(echo "$f" | cut -f1)" -gt 8192 ] && log "note: $arm b$blk request $r re-evaluated $(echo "$f" | cut -f1) prompt tokens (the slot did not keep the text)"
  done
  kpid $SPID; wait $SPID 2>/dev/null; SPID=""; sleep 3
  if grep -qiE 'NCCL init failed|probe AllReduce failed' $O/$arm-$blk-server.log; then
    log "RCCL NOT UP in $arm b$blk -- discarded"; printf "mtp\t%s\t%s\tNORCCL\tNORCCL\n" "$arm" "$blk" >> $TSV; return 0
  fi
  if [ $ok = 0 ]; then printf "mtp\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$blk" >> $TSV; return 0; fi
  mean=$(echo $dec | tr ' ' '\n' | awk '{ s += $1; n++ } END { printf "%.4f", s / n }')
  printf "mtp\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$blk" "$pre" "$mean" "$(echo $acc)" "$(echo $sha)" "$(echo $dec)" >> $TSV
  log "$arm b$blk: prefill $pre tok/s ($pn tokens); MTP decode mean $mean ($(echo $dec)); accepted $(echo $acc)"
}
for blk in $(seq 1 $N); do
  order=$(echo $ARMS | tr ' ' '\n' | shuf | tr '\n' ' '); log "block $blk order: $order"
  for arm in $order; do cell $arm $blk; done
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore; touch $DONE; log "=== SUB-MTP $TAG DONE ==="
