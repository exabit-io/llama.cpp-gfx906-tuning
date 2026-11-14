#!/bin/bash
# queue2.sh -- v1.0.0 cycle, after queue1:
#   1. which MTP row path really runs (-v logs, not timed): combined, dev, host
#   2. stock v1.0.0: test-backend-ops TOPK_MOE (the wave64 ballot bug) on one die
#   3. the deb: reference on the previous package, install, suite, MTP
#   4. DeepSeek-V4.1 UD-IQ1_S: stock (layer, ppl), merge build (full)
#   5. Nemotron 3.5 Bolt: stock, merge build, master
set -u
W=/root/rebase-v100
PR=/root/build-pr-v100; MA=/root/build-master-v100; ST=/root/build-stock-v100
Q=$W/queue2.progress
M=/root/models/Qwen3.8-27B-Q8_0.gguf
log(){ echo "$(date -Is) [queue2] $*" | tee -a $Q; }
until grep -q 'QUEUE1 DONE' $W/queue1.progress 2>/dev/null; do sleep 20; done
log "start"
. /root/rocm-tests/bench/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
O=$W/mtp-rows-diag; mkdir -p $O
setmax
for arm in combined dev host; do
  env_arm=""
  [ $arm = host ] && env_arm="LLAMA_MTP_DEV=0"
  [ $arm = dev  ] && env_arm="LLAMA_MTP_DEFERRED_PREFILL=0"
  env $env_arm LD_LIBRARY_PATH=$PR/bin:/opt/rocm/lib $PR/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -c 8192 -np 1 --port 8097 \
    --spec-type draft-mtp --spec-draft-n-max 2 -v > $O/$arm-server.out 2> $O/$arm-server.log < /dev/null &
  S=$!
  if wait_server 8097 $S; then
    curl -s --max-time 600 http://127.0.0.1:8097/completion -H 'Content-Type: application/json' -d @$W/mtp-modes-pr/req-short.json > $O/$arm.json
    log "rows $arm: $(grep -c 'h_nextn buffer size' $O/$arm-server.log) device buffer allocations ($(grep -m1 -o 'h_nextn buffer size = *[0-9.]* MiB' $O/$arm-server.log)), $(grep -c 'extracting the h_nextn rows to host' $O/$arm-server.log) fallbacks, $(grep -c 'sched_reserve: reserving' $O/$arm-server.log) reserves; $(python3 -I -c 'import json,sys,hashlib; d=json.load(open(sys.argv[1])); t=d["timings"]; print("accepted %s/%s sha %s" % (t.get("draft_n_accepted"), t.get("draft_n"), hashlib.sha256(d["content"].encode()).hexdigest()[:12]))' $O/$arm.json 2>&1)"
  else
    log "rows $arm: server did not come up"
  fi
  kill $S; wait $S 2>/dev/null; sleep 3
done
restore
bash $W/ctx-check.sh $PR $W/ctx-pr > $W/ctx-pr.out 2>&1; log "ctx-check merge rc=$?"
bash $W/ctx-check.sh $ST $W/ctx-stock > $W/ctx-stock.out 2>&1; log "ctx-check stock rc=$?"
bash $W/ctx-check.sh $MA $W/ctx-master > $W/ctx-master.out 2>&1; log "ctx-check master rc=$?"
O=$W/stock-topk; mkdir -p $O
LD_LIBRARY_PATH=$ST/bin:/opt/rocm/lib timeout 1800 $ST/bin/test-backend-ops -b ROCm0 -o TOPK_MOE > $O/tbo-0.log 2>&1
log "stock v1.0.0 TOPK_MOE ROCm0: rc=$? $(grep -E '[0-9]+/[0-9]+ tests passed' $O/tbo-0.log | tail -1); FAIL lines $(grep -c 'FAIL' $O/tbo-0.log)"
LD_LIBRARY_PATH=$PR/bin:/opt/rocm/lib timeout 1800 $PR/bin/test-backend-ops -b ROCm0 -o TOPK_MOE > $O/tbo-pr-0.log 2>&1
log "merge TOPK_MOE ROCm0: rc=$? $(grep -E '[0-9]+/[0-9]+ tests passed' $O/tbo-pr-0.log | tail -1); FAIL lines $(grep -c 'FAIL' $O/tbo-pr-0.log)"
# build-mx-llama-deb.sh writes last-deb.txt as its last step
until grep -q '_1\.0\.0\.git' /root/mx-llama-deb/last-deb.txt 2>/dev/null && [ -f "$(cat /root/mx-llama-deb/last-deb.txt)" ]; do sleep 10; done
bash $W/deb-test.sh $W/deb-test > $W/deb-test.out 2>&1; log "deb-test rc=$?"
bash $W/dsv41-test.sh $ST $W/dsv41-stock > $W/dsv41-stock.out 2>&1; log "dsv41 stock rc=$?"
bash $W/dsv41-test.sh $PR $W/dsv41-pr full > $W/dsv41-pr.out 2>&1; log "dsv41 merge rc=$?"
bash $W/nemotron-test.sh $ST $W/nemotron-stock > $W/nemotron-stock.out 2>&1; log "nemotron stock rc=$?"
bash $W/nemotron-test.sh $PR $W/nemotron-pr > $W/nemotron-pr.out 2>&1; log "nemotron merge rc=$?"
bash $W/nemotron-test.sh $MA $W/nemotron-master > $W/nemotron-master.out 2>&1; log "nemotron master rc=$?"
log "=== QUEUE2 DONE ==="
