#!/bin/bash
# mtp-deb.sh -- MTP speculative decode on the installed deb, idle host (the PR suite's step 4, 3 requests)
set -u
R=/root/rocm-tests/bench; O=/root/rebase-v070/mtp-deb; mkdir -p $O
M=/root/models/Qwen3.8-27B-Q8_0.gguf; B=/opt/mx-llama.cpp
. $R/gpu-test-env.sh; setmax
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml HSA_FORCE_FINE_GRAIN_PCIE=1
$B/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -c 8192 -np 1 --port 8092 \
  --spec-type draft-mtp --spec-draft-n-max 2 > $O/server.out 2> $O/server.log < /dev/null &
SPID=$!
trap 'kill $SPID 2>/dev/null; restore' EXIT
if wait_server 8092 $SPID; then
  for i in 1 2 3; do
    curl -s --max-time 600 http://127.0.0.1:8092/completion -H 'Content-Type: application/json' \
      -d '{"prompt":"Write a short paragraph about the history of the transistor.","n_predict":256,"temperature":0,"top_k":1,"seed":0}' > $O/r$i.json
    python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); t=d["timings"]; print("run", sys.argv[2], "tokens", t["predicted_n"], "tok/s %.2f" % t["predicted_per_second"], "draft", t.get("draft_n"), "accepted", t.get("draft_n_accepted"))' $O/r$i.json $i
  done
else
  echo "server did not come up: $(tail -3 $O/server.log)"
fi
