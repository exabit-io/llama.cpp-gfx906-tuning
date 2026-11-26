#!/bin/bash
# rs-modes2.sh BUILD OUTDIR -- second session of the rollback question, after rs-modes.sh found that replay and the
# plane-indexed snapshots give different text at -np 4. All arms -np 4, MTP n_max 3, -sm tensor, one request at a time:
#   replay   default: upstream's replay; every gdn op carries replay inputs and runs on the RECURRENT kernel
#   snap     LLAMA_RS_REPLAY=0: snapshots, gdn ops on the fork's column-thread / chunked kernels
#   snaprec  LLAMA_RS_REPLAY=0 GGML_GDN_NO_COLTHREAD=1 LLAMA_GDN_NO_CHUNK_SNAPSHOTS=1: snapshots, every gdn op on the
#            recurrent kernel as well
#   nodraft  no drafter (no rollback ever happens): the text of plain decode, for the record
# READING, written before the first run. The state after a rollback must be the state the tokens kept would have
# produced. replay and snaprec differ ONLY in how they roll back (restore a snapshot / replay the kept tokens), so:
# (1) replay and snaprec give the same text on all four requests -> both rollbacks agree, and the difference rs-modes.sh
#     saw is the other gdn kernel of the snapshot arm (rounding), not a fault; replay may then be the default.
# (2) they differ -> not decided by text: one rollback is off by more than rounding or the two variants of the recurrent
#     kernel round differently; the state-level test (test-llama-context: rollback, rollback_bound, replay) decides, and
#     until it does parallel contexts stay on snapshots (LLAMA_RS_REPLAY default off in this fork).
# (3) snap against snaprec shows what the kernel alone changes. nodraft is not required to equal any arm: drafts change
#     the verify batch shapes, and other batch shapes run other kernels.
# The mode of each arm is read from the log (-lv 4): "rs_replay = true|false" and the recurrent memory's size line.
set -u
B=$1; O=$2
R=/root/rocm-tests/bench
M=/root/models/Qwen3.8-27B-Q8_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
mkdir -p $O; P=$O/progress
LDP=$B/bin:$B/lib:${RT:-/opt/rocm/core-10.1}/lib
log(){ echo "$(date -Is) [rs-modes2] $*" | tee -a $P; }
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
for arm in ${ARMS:-replay snap snaprec nodraft}; do
  env_arm="LLAMA_RS_DUMMY=0"; NP=4; SPEC="--spec-type draft-mtp --spec-draft-n-max ${NMAX:-3}"
  [ $arm = snap    ] && env_arm="LLAMA_RS_REPLAY=0"
  [ $arm = snaprec ] && env_arm="LLAMA_RS_REPLAY=0 GGML_GDN_NO_COLTHREAD=1 LLAMA_GDN_NO_CHUNK_SNAPSHOTS=1"
  [ $arm = nodraft ] && SPEC=""
  env $env_arm LD_LIBRARY_PATH=$LDP $B/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -c $((8192 * NP)) -np $NP --port 8098 \
    $SPEC -lv 4 > $O/$arm-server.out 2> $O/$arm-server.log < /dev/null &
  S=$!
  if wait_server 8098 $S; then
    curl -s --max-time 900 http://127.0.0.1:8098/completion -H 'Content-Type: application/json' -d @$O/req-long.json > $O/$arm-long.json
    log "$arm long : $(show $O/$arm-long.json)"
    for r in 1 2 3; do
      curl -s --max-time 600 http://127.0.0.1:8098/completion -H 'Content-Type: application/json' -d @$O/req-short.json > $O/$arm-$r.json
      log "$arm run $r: $(show $O/$arm-$r.json)"
    done
    log "$arm mode : rs_replay=$(grep -m1 -oE 'rs_replay += [a-z]+' $O/$arm-server.log | awk '{print $3}'), ring lines $(grep -c 'snapshot ring enabled' $O/$arm-server.log), refused rollbacks $(grep -c 'refusing rollback' $O/$arm-server.log); $(grep -m1 -E 'llama_memory_recurrent: size' $O/$arm-server.log | sed 's/.*size = //' | cut -c1-80)"
  else
    log "$arm: server did not come up: $(tail -3 $O/$arm-server.log | tr '\n' ' ' | cut -c1-300)"
  fi
  kill $S; wait $S 2>/dev/null; S=""; sleep 3
done
sha(){ python3 -I -c 'import json,sys,hashlib; print(hashlib.sha256(json.load(open(sys.argv[1]))["content"].encode()).hexdigest()[:12])' $1 2>/dev/null || echo none; }
for r in long 1 2 3; do
  log "text $r: replay $(sha $O/replay-$r.json) snaprec $(sha $O/snaprec-$r.json) snap $(sha $O/snap-$r.json) nodraft $(sha $O/nodraft-$r.json)"
done
log "=== RS-MODES2 DONE ==="
