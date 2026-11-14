#!/bin/bash
# mtp-modes.sh BUILD OUTDIR [NP] -- MTP target hidden rows: where the draft context reads them from.
#   combined  default: deferred prefill of the first prompt (host accum buffer), device rows after its replay
#   host      LLAMA_MTP_DEV=0: deferred prefill, host rows (the fork before v1.0.0)
#   dev       LLAMA_MTP_DEFERRED_PREFILL=0: device rows from the start (upstream v1.0.0)
#   eager     both off: host rows, per-chunk catch-up
# Per arm: one long-prompt request (prefill), then three identical short greedy requests (decode), 256 tokens, -sm tensor.
# Greedy output must be identical in all arms: the rows are the same values on every path.
set -u
B=$1; O=$2; NP=${3:-1}
R=/root/rocm-tests/bench
M=/root/models/Qwen3.8-27B-Q8_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
mkdir -p $O; P=$O/progress
LDP=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib
log(){ echo "$(date -Is) [mtp-modes] $*" | tee -a $P; }
. $R/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
S=""
trap 'kill $S 2>/dev/null; restore' EXIT
setmax
python3 -I - "$F" "$O" <<'EOF'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()[:9000]
short = {"prompt": "Write a short paragraph about the history of the transistor.", "n_predict": 256, "temperature": 0, "top_k": 1, "seed": 0, "cache_prompt": False}
long_ = dict(short, prompt=text + "\n\nSummarize the text above in one paragraph.\n")
json.dump(short, open(sys.argv[2] + "/req-short.json", "w"))
json.dump(long_, open(sys.argv[2] + "/req-long.json", "w"))
EOF
show(){ python3 -I -c 'import json,sys,hashlib; d=json.load(open(sys.argv[1])); t=d["timings"]; print("prompt %d tok %.1f tok/s | gen %d tok %.2f tok/s | accepted %s/%s | sha %s" % (t["prompt_n"], t["prompt_per_second"], t["predicted_n"], t["predicted_per_second"], t.get("draft_n_accepted"), t.get("draft_n"), hashlib.sha256(d["content"].encode()).hexdigest()[:12]))' "$1" 2>&1; }
for arm in ${ARMS:-combined host dev eager}; do
  env_arm=""
  [ $arm = host  ] && env_arm="LLAMA_MTP_DEV=0"
  [ $arm = dev   ] && env_arm="LLAMA_MTP_DEFERRED_PREFILL=0"
  [ $arm = eager ] && env_arm="LLAMA_MTP_DEV=0 LLAMA_MTP_DEFERRED_PREFILL=0"
  env $env_arm LD_LIBRARY_PATH=$LDP $B/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -c $((8192 * NP)) -np $NP --port 8098 \
    --spec-type draft-mtp --spec-draft-n-max 2 -lv 3 > $O/$arm-server.out 2> $O/$arm-server.log < /dev/null &
  S=$!
  if wait_server 8098 $S; then
    curl -s --max-time 900 http://127.0.0.1:8098/completion -H 'Content-Type: application/json' -d @$O/req-long.json > $O/$arm-long.json
    log "$arm long : $(show $O/$arm-long.json)"
    for r in 1 2 3; do
      curl -s --max-time 600 http://127.0.0.1:8098/completion -H 'Content-Type: application/json' -d @$O/req-short.json > $O/$arm-$r.json
      log "$arm run $r: $(show $O/$arm-$r.json)"
    done
    log "$arm rows : $(grep -c 'h_nextn buffer size' $O/$arm-server.log) device buffer allocations, $(grep -c 'extracting the h_nextn rows to host' $O/$arm-server.log) fallbacks, $(grep -c 'reserving \.\.\.' $O/$arm-server.log) reserves, $(grep -ciE 'failed|error|abort' $O/$arm-server.log) error lines"
  else
    log "$arm: server did not come up: $(tail -3 $O/$arm-server.log | tr '\n' ' ' | cut -c1-300)"
  fi
  kill $S; wait $S 2>/dev/null; S=""; sleep 3
done
log "=== MTP-MODES DONE ==="
