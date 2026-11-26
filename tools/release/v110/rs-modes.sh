#!/bin/bash
# rs-modes.sh BUILD OUTDIR -- the rollback of the recurrent (delta-net) state under MTP drafts, v1.1.0 merge:
#   ring    -np 1: the fork's snapshot ring (unchanged by the merge)
#   replay  -np 4, default: upstream's replay (#31669), the gdn op replays the kept pending tokens
#   snap    -np 4, LLAMA_RS_REPLAY=0: the plane-indexed snapshots the fork used for parallel contexts before the merge
#   ringoff -np 1, LLAMA_RS_REPLAY=0: must be the ring again (the switch only concerns contexts the ring does not cover)
# Per arm: one long-prompt request (prefill), then three identical short greedy requests (decode), 256 tokens, -sm tensor,
# one request at a time. READING, written before the first run: a rejected draft token rolls the state back, so a wrong
# rollback shows as different text. (1) replay and snap must give the SAME text for the long request and for each short
# request (same shapes, same kernels except the rollback); (2) ring and ringoff must be identical to each other and say
# "snapshot ring enabled" in the log; (3) no arm may log an error line or a rollback refusal. ring against the -np 4
# arms is reported, not required (other batch shapes). Speeds are per arm of one session: prefill of the long request
# and decode of the short ones, replay against snap, is the cost or gain of following upstream for parallel contexts.
set -u
B=$1; O=$2
R=/root/rocm-tests/bench
M=/root/models/Qwen3.8-27B-Q8_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
mkdir -p $O; P=$O/progress
LDP=$B/bin:$B/lib:${RT:-/opt/rocm/core-10.1}/lib
log(){ echo "$(date -Is) [rs-modes] $*" | tee -a $P; }
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
for arm in ${ARMS:-ring replay snap ringoff}; do
  env_arm="LLAMA_RS_DUMMY=0"; NP=4
  [ $arm = ring    ] && NP=1
  [ $arm = snap    ] && env_arm="LLAMA_RS_REPLAY=0"
  [ $arm = ringoff ] && { NP=1; env_arm="LLAMA_RS_REPLAY=0"; }
  env $env_arm LD_LIBRARY_PATH=$LDP $B/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -c $((8192 * NP)) -np $NP --port 8098 \
    --spec-type draft-mtp --spec-draft-n-max ${NMAX:-3} -lv 3 > $O/$arm-server.out 2> $O/$arm-server.log < /dev/null &
  S=$!
  if wait_server 8098 $S; then
    curl -s --max-time 900 http://127.0.0.1:8098/completion -H 'Content-Type: application/json' -d @$O/req-long.json > $O/$arm-long.json
    log "$arm long : $(show $O/$arm-long.json)"
    for r in 1 2 3; do
      curl -s --max-time 600 http://127.0.0.1:8098/completion -H 'Content-Type: application/json' -d @$O/req-short.json > $O/$arm-$r.json
      log "$arm run $r: $(show $O/$arm-$r.json)"
    done
    log "$arm mode : rs_replay=$(grep -m1 -oE 'rs_replay += [a-z]+' $O/$arm-server.log | awk '{print $3}'), ring lines $(grep -c 'snapshot ring enabled' $O/$arm-server.log), refused rollbacks $(grep -c 'refusing rollback' $O/$arm-server.log), $(grep -c 'reserving \.\.\.' $O/$arm-server.log) reserves, $(grep -ciE 'failed|error|abort' $O/$arm-server.log) error lines; $(grep -m1 -E 'llama_memory_recurrent: size' $O/$arm-server.log | sed 's/.*size = //' | cut -c1-90)"
  else
    log "$arm: server did not come up: $(tail -3 $O/$arm-server.log | tr '\n' ' ' | cut -c1-300)"
  fi
  kill $S; wait $S 2>/dev/null; S=""; sleep 3
done
sha(){ python3 -I -c 'import json,sys,hashlib; print(hashlib.sha256(json.load(open(sys.argv[1]))["content"].encode()).hexdigest()[:12])' $1 2>/dev/null || echo none; }
for r in long 1 2 3; do
  log "text $r: ring $(sha $O/ring-$r.json) ringoff $(sha $O/ringoff-$r.json) replay $(sha $O/replay-$r.json) snap $(sha $O/snap-$r.json)"
done
log "=== RS-MODES DONE ==="
