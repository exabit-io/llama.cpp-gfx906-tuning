#!/bin/bash
# lilicorr-ab.sh -- Nemotron 3.5 Bolt + LiLiCorr drafter, four prompts, greedy 192 tokens, stock vs PR build (one die)
set -u
N=/root/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
M=$N/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf; L=$N/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf
O=/root/rebase-v090/lilicorr-ab; mkdir -p $O
. /root/rocm-tests/bench/gpu-test-env.sh; setmax
for b in stock:/root/build-stock-v090 pr:/root/build-pr-v090; do
  n=${b%%:*}; B=${b#*:}
  LD_LIBRARY_PATH=$B/bin:/opt/rocm/lib $B/bin/llama-server -m $M --device rocm0 -ngl all -fa on -c 16384 -b 2048 -ub 512 -np 1 --port 8097 \
    --spec-type draft-dflash --spec-draft-n-max 6 --spec-draft-model $L --spec-draft-ngl 999 --spec-draft-device rocm0 > $O/$n-server.out 2> $O/$n-server.log < /dev/null &
  S=$!
  if wait_server 8097 $S; then
    i=0
    for q in "Write a short paragraph about the history of the transistor." "Explain how a refrigerator works, step by step." \
             "Write a Python function that returns the n-th Fibonacci number, with a docstring." "List five uses of copper and why each needs copper."; do
      curl -s --max-time 600 http://127.0.0.1:8097/completion -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$q\",\"n_predict\":192,\"temperature\":0,\"top_k\":1,\"seed\":0}" > $O/$n-$i.json
      python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); t=d["timings"]; print(sys.argv[2], "prompt", sys.argv[3], "tokens", t["predicted_n"], "accepted %s/%s"%(t.get("draft_n_accepted"),t.get("draft_n")), "rate %.0f%%"%(100*t.get("draft_n_accepted",0)/max(1,t.get("draft_n",1))), "tok/s %.1f"%t["predicted_per_second"])' $O/$n-$i.json $n $i
      i=$((i+1))
    done
  else
    echo "$n: server did not come up"
  fi
  kill $S; wait $S 2>/dev/null; sleep 3
done
restore
