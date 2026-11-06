#!/bin/bash
# mtp-ab.sh BUILD OUTDIR -- MTP decode rate, default (chunked CUDA graphs, #30804) vs GGML_CUDA_GRAPH_CHUNK_SIZE=0
# (one graph per ggml graph, as before v0.8.0). Three identical greedy requests per arm, 256 tokens, -sm tensor.
set -u
B=$1; O=$2
R=/root/rocm-tests/bench
M=/root/models/Qwen3.8-27B-Q8_0.gguf
mkdir -p $O; P=$O/progress
LDP=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib
log(){ echo "$(date -Is) [mtp-ab] $*" | tee -a $P; }
. $R/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
setmax
for arm in default chunk0; do
  env_arm=""; [ $arm = chunk0 ] && env_arm="GGML_CUDA_GRAPH_CHUNK_SIZE=0"
  env $env_arm LD_LIBRARY_PATH=$LDP $B/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -c 8192 -np 1 --port 8098 \
    --spec-type draft-mtp --spec-draft-n-max 2 > $O/$arm-server.out 2> $O/$arm-server.log < /dev/null &
  S=$!
  if wait_server 8098 $S; then
    for r in 1 2 3; do
      curl -s --max-time 600 http://127.0.0.1:8098/completion -H 'Content-Type: application/json' \
        -d '{"prompt":"Write a short paragraph about the history of the transistor.","n_predict":256,"temperature":0,"top_k":1,"seed":0,"cache_prompt":false}' > $O/$arm-$r.json
      log "$arm run $r: $(python3 -I -c 'import json,sys; t=json.load(open(sys.argv[1]))["timings"]; print("tok/s %.2f"%t["predicted_per_second"], "accepted %s/%s"%(t.get("draft_n_accepted"),t.get("draft_n")))' $O/$arm-$r.json)"
    done
  else
    log "$arm: server did not come up"
  fi
  kill $S; wait $S 2>/dev/null; sleep 3
done
restore
log "=== MTP-AB DONE ==="
