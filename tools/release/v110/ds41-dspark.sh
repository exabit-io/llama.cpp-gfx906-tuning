#!/bin/bash
# ds41-dspark.sh BUILD OUTDIR -- the DSpark drafter on DeepSeek-V4.1 (upstream-layout file) with the port: llama-server
# -sm layer, -fitt 7168 so that the drafter has room (dsv41-test.sh's step 4 has no -fitt and cannot load it), greedy
# 192 tokens without and with --spec-type draft-dspark. Correctness only (dies at 1000 MHz): no rate is a number.
# READING, before the run: the drafter loads and the request completes with drafted tokens > 0. The 1.0.0 package had
# 115/227 accepted with upstream's drafter graph; the port drafts through the fork's graph, so the count is reported
# and not judged. A load failure or 0 drafted tokens = the drafter does not work on this layout: say so in the notes.
set -u
B=$1; O=$2; RT=${RT:-/opt/rocm/core-10.0}; mkdir -p $O; P=$O/progress
D=/root/models/DeepSeek-V4.1-Flash-1017-UD-IQ1_S
log(){ echo "$(date -Is) [ds41-dspark] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is running"; exit 1; }
. /root/rocm-tests/bench/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
S=""
trap 'kill $S 2>/dev/null; restore' EXIT
setmax
for mode in base dspark; do
  extra=""; [ $mode = dspark ] && extra="--spec-type draft-dspark -md $D/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"
  LD_LIBRARY_PATH=$B/bin:$RT/lib $B/bin/llama-server -m $D/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf --device rocm0,rocm1,rocm2,rocm3 -sm layer -fa on -c 8192 -np 1 -fitt 7168 --port 8100 $extra > $O/$mode.out 2> $O/$mode.log < /dev/null &
  S=$!
  if wait_server 8100 $S; then
    curl -s --max-time 1800 http://127.0.0.1:8100/completion -H 'Content-Type: application/json' -d '{"prompt":"The transistor was invented in 1947 at Bell Labs. In the decades that followed,","n_predict":192,"temperature":0,"top_k":1,"seed":0,"ignore_eos":true,"cache_prompt":false}' > $O/$mode.json
    log "$mode: $(python3 -I -c 'import json,sys,hashlib; d=json.load(open(sys.argv[1])); t=d["timings"]; print("tokens %s, drafted %s, accepted %s, text md5 %s: %s" % (t.get("predicted_n"), t.get("draft_n"), t.get("draft_n_accepted"), hashlib.md5(d["content"].encode()).hexdigest()[:8], d["content"][:110].replace("\n"," ")))' $O/$mode.json 2>&1 | tail -1)"
  else
    log "$mode: server did not come up: $(grep -a -E ' E |error|failed' $O/$mode.log | tail -3 | tr '\n' ' ' | cut -c1-400)"
  fi
  kill $S 2>/dev/null; wait $S 2>/dev/null; S=""; sleep 5
done
log "drafter path lines: $(grep -a -h -E 'deferred prompt encode|target features via' $O/dspark.log | sed -E 's/^[0-9. ]*[A-Z] //' | sort | uniq -c | tr '\n' ';' | cut -c1-300)"
log "=== DS41-DSPARK DONE ==="
