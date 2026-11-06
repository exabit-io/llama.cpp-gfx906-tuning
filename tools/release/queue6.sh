#!/bin/bash
# queue6.sh -- after queue5: DeepSeek-V4.1 DSpark with room for the drafter (the fit cannot measure a DFlash/DSpark
# drafter, so it fills the dies with the target; --fit-target 7168 leaves 7 GiB per die), on stock and the PR build,
# plus stock -sm layer greedy with the continuation prompt the PR run used.
set -u
W=/root/rebase-v090
Q=$W/queue6.progress
log(){ echo "$(date -Is) [queue6] $*" | tee -a $Q; }
until grep -q 'QUEUE5 DONE' $W/queue5.progress 2>/dev/null; do sleep 20; done
log "start"
. /root/rocm-tests/bench/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
D=/root/models/DeepSeek-V4.1-Flash-1017-UD-IQ1_S
M=$D/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf; DS=$D/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
D4="--device rocm0,rocm1,rocm2,rocm3"
PROMPT="The transistor was invented in 1947 at Bell Labs. In the decades that followed,"
O=$W/dsv41-dspark; mkdir -p $O
setmax
LD_LIBRARY_PATH=/root/build-stock-v090/bin:/opt/rocm/lib timeout 3600 /root/build-stock-v090/bin/llama-completion -m $M $D4 -sm layer -fa on -c 4096 -n 64 \
  --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup -p "$PROMPT" > $O/stock-layer.out 2> $O/stock-layer.log < /dev/null
log "stock -sm layer greedy: rc=$? $(wc -c < $O/stock-layer.out) bytes, md5 $(md5sum < $O/stock-layer.out | cut -c1-8): $(tr '\n' ' ' < $O/stock-layer.out | cut -c1-160)"
log "  speed: $(grep -E '       eval time' $O/stock-layer.log | tail -1 | sed 's/.*= *//' | cut -c1-120)"
for b in stock:/root/build-stock-v090 pr:/root/build-pr-v090; do
  n=${b%%:*}; B=${b#*:}
  for mode in base dspark; do
    extra=""; [ $mode = dspark ] && extra="--spec-type draft-dspark -md $DS"
    LD_LIBRARY_PATH=$B/bin:/opt/rocm/lib $B/bin/llama-server -m $M $D4 -sm layer -fa on -c 8192 -np 1 -fitt 7168 --port 8099 $extra \
      > $O/$n-$mode.out 2> $O/$n-$mode.log < /dev/null &
    S=$!
    if wait_server 8099 $S; then
      curl -s --max-time 1800 http://127.0.0.1:8099/completion -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":192,\"temperature\":0,\"top_k\":1,\"seed\":0}" > $O/$n-$mode.json
      log "$n $mode: $(python3 -I -c 'import json,sys,hashlib; d=json.load(open(sys.argv[1])); t=d.get("timings",{}); print("tokens", t.get("predicted_n"), "tok/s %.2f"%t.get("predicted_per_second",0), "draft", t.get("draft_n_accepted"), "/", t.get("draft_n"), "| md5", hashlib.md5(d.get("content","").encode()).hexdigest()[:8], "|", d.get("content","")[:90].replace(chr(10)," "))' $O/$n-$mode.json 2>&1)"
    else
      log "$n $mode: server did not come up: $(grep -E ' E ' $O/$n-$mode.log | tail -2 | tr '\n' ' ' | cut -c1-260)"
    fi
    kill $S 2>/dev/null; wait $S 2>/dev/null; sleep 5
  done
done
restore
log "=== QUEUE6 DONE ==="
