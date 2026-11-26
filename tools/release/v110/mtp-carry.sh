#!/bin/bash
# mtp-carry.sh BUILD OUTDIR -- does a request's MTP draft depend on the request that ran before it in the slot?
# The first token of a sequence is paired with a "carried" hidden row; a sequence that starts at position 0 has none and
# gets zeros. In mxxm-t's tree (and in the v1.1.0 merge) the zeroing runs after the row was uploaded to the device buffer,
# so with device rows a request processed from position 0 is paired with the LAST row of the slot's previous request.
# One server (Qwen3.8-27B Q8_0, -sm tensor, -np 1, MTP n_max 2, cache_prompt false, greedy), requests in this order:
#   A (fresh slot: the carried row is the constructor's zeros), A, A, B, A
# READING, written before the first run: the text is the target's and must be the same for every A. With a correct
# reset the draft statistics (accepted / drafted) of every A are the same too. If the first A differs from the later
# ones, or the A after B differs from the A after A, the draft depends on the previous request.
set -u
B=$1; O=$2
R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
mkdir -p $O; P=$O/progress
LDP=$B/bin:$B/lib:${RT:-/opt/rocm/core-10.1}/lib
log(){ echo "$(date -Is) [mtp-carry] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
. $R/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
S=""
trap 'kill $S 2>/dev/null; restore' EXIT
setmax
python3 -I - "$O" <<'EOF2'
import json, sys
base = {"n_predict": 256, "temperature": 0, "top_k": 1, "seed": 0, "cache_prompt": False}
json.dump(dict(base, prompt="Write a short paragraph about the history of the transistor."), open(sys.argv[1] + "/req-A.json", "w"))
json.dump(dict(base, prompt="Explain in one paragraph how a refrigerator moves heat from its inside to the room."), open(sys.argv[1] + "/req-B.json", "w"))
EOF2
show(){ python3 -I -c 'import json,sys,hashlib; d=json.load(open(sys.argv[1])); t=d["timings"]; print("gen %d tok | accepted %s/%s | sha %s" % (t["predicted_n"], t.get("draft_n_accepted"), t.get("draft_n"), hashlib.sha256(d["content"].encode()).hexdigest()[:12]))' $1; }
env ${ENVARM:-LLAMA_CARRY_DUMMY=0} LD_LIBRARY_PATH=$LDP $B/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -c 8192 -np 1 --port 8098 \
  --spec-type draft-mtp --spec-draft-n-max 2 > $O/server.out 2> $O/server.log < /dev/null &
S=$!
if wait_server 8098 $S; then
  i=0
  for r in A A A B A; do
    i=$((i+1))
    curl -s --max-time 600 http://127.0.0.1:8098/completion -H 'Content-Type: application/json' -d @$O/req-$r.json > $O/$i-$r.json
    log "$(basename $B) request $i ($r): $(show $O/$i-$r.json)"
  done
else
  log "server did not come up: $(tail -3 $O/server.log | tr '\n' ' ' | cut -c1-300)"
fi
kill $S; wait $S 2>/dev/null; S=""
log "=== MTP-CARRY DONE ==="
