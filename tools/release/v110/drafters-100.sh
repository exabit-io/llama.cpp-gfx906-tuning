#!/bin/bash
# drafters-100.sh OUT -- the two drafter models on v1.0.0 (the v0.9.0 cycle's dspark-ab.sh and lilicorr-ab.sh, one script):
#  1. DeepSeek-V4.1 + DSpark sidecar, -sm layer over four dies, --fit-target 7168 (without it the fit fills the dies with
#     the target and the drafter cannot allocate): stock and merge build, without and with the drafter; on the merge build
#     also the two other feature paths (LLAMA_DFLASH_DEFER=0: upstream's device features; + LLAMA_DFLASH_DEV=0: host copies).
#     Two prompts, greedy, 192 tokens, ignore_eos.
#  2. Nemotron 3.5 Bolt + LiLiCorr drafter (draft-dflash), one die: stock, merge build, master; four prompts, greedy, 192.
# Waits for np4-ab.sh (the GPUs are its until then). Smoke depths: build to build only.
set -u
W=/root/rebase-v100; O=$1; mkdir -p $O; P=$O/progress
ST=/root/build-stock-v100; PR=/root/build-pr-v100; MA=/root/build-master-v100
D=/root/models/DeepSeek-V4.1-Flash-1017-UD-IQ1_S
DM=$D/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf; DS=$D/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
N=/root/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
NM=$N/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf; NL=$N/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf
log(){ echo "$(date -Is) [drafters] $*" | tee -a $P; }
until grep -q 'NP4-AB DONE' $W/np4-ab/progress 2>/dev/null; do sleep 15; done
for f in $DM $DS $NM $NL; do [ -f $f ] || { log "missing $f"; exit 1; }; done
. /root/rocm-tests/bench/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
S=""
trap 'kill $S 2>/dev/null; restore' EXIT
setmax
python3 -I - $O <<'EOF'
import json, sys
o = sys.argv[1]
ds = ["The transistor was invented in 1947 at Bell Labs. In the decades that followed,",
      "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"]
for i, q in enumerate(ds):
    json.dump({"prompt": q, "n_predict": 192, "temperature": 0, "top_k": 1, "seed": 0, "ignore_eos": True, "cache_prompt": False}, open("%s/ds-req-%d.json" % (o, i), "w"))
nm = ["Write a short paragraph about the history of the transistor.", "Explain how a refrigerator works, step by step.",
      "Write a Python function that returns the n-th Fibonacci number, with a docstring.", "List five uses of copper and why each needs copper."]
for i, q in enumerate(nm):
    json.dump({"prompt": q, "n_predict": 192, "temperature": 0, "top_k": 1, "seed": 0, "cache_prompt": False}, open("%s/nm-req-%d.json" % (o, i), "w"))
EOF
show(){ python3 -I -c 'import json,sys,hashlib; d=json.load(open(sys.argv[1])); t=d["timings"]; print("tokens %s tok/s %.2f accepted %s/%s md5 %s" % (t["predicted_n"], t["predicted_per_second"], t.get("draft_n_accepted"), t.get("draft_n"), hashlib.md5(d["content"].encode()).hexdigest()[:8]))' "$1" 2>&1 | tail -1; }

# ds NAME BUILD ENV DRAFTER(0|1)
ds(){ local n=$1 B=$2 e=$3 dr=$4 extra=""
  [ $dr = 1 ] && extra="--spec-type draft-dspark -md $DS"
  env $e LD_LIBRARY_PATH=$B/bin:${RT:-/opt/rocm/core-10.1}/lib $B/bin/llama-server -m $DM --device rocm0,rocm1,rocm2,rocm3 -sm layer -fa on \
    -c 8192 -np 1 -fitt 7168 --port 8100 $extra > $O/ds-$n.out 2> $O/ds-$n.log < /dev/null &
  S=$!
  if wait_server 8100 $S; then
    [ $dr = 1 ] && log "ds $n path: $(grep -hE 'deferred prompt encode|target features via' $O/ds-$n.log | sed 's/^[0-9.]* I //; s/common_speculative_impl_draft_dflash: //' | tr '\n' ';' | cut -c1-160)"
    for i in 0 1; do
      curl -s --max-time 1800 http://127.0.0.1:8100/completion -H 'Content-Type: application/json' -d @$O/ds-req-$i.json > $O/ds-$n-$i.json
      log "ds $n prompt $i: $(show $O/ds-$n-$i.json)"
    done
  else
    log "ds $n: server did not come up: $(grep -E ' E ' $O/ds-$n.log | tail -2 | tr '\n' ' ' | cut -c1-240)"
  fi
  kill $S; wait $S 2>/dev/null; S=""; sleep 5
}
ds stock-base   $ST "" 0
ds stock-dspark $ST "" 1
ds pr-base      $PR "" 0
ds pr-deferred  $PR "" 1
ds pr-device    $PR "LLAMA_DFLASH_DEFER=0" 1
ds pr-host      $PR "LLAMA_DFLASH_DEFER=0 LLAMA_DFLASH_DEV=0" 1

for b in stock:$ST pr:$PR master:$MA; do
  n=${b%%:*}; B=${b#*:}
  LD_LIBRARY_PATH=$B/bin:${RT:-/opt/rocm/core-10.1}/lib $B/bin/llama-server -m $NM --device rocm0 -ngl all -fa on -c 16384 -b 2048 -ub 512 -np 1 --port 8097 \
    --spec-type draft-dflash --spec-draft-n-max 6 --spec-draft-model $NL --spec-draft-ngl 999 --spec-draft-device rocm0 > $O/nm-$n.out 2> $O/nm-$n.log < /dev/null &
  S=$!
  if wait_server 8097 $S; then
    for i in 0 1 2 3; do
      curl -s --max-time 600 http://127.0.0.1:8097/completion -H 'Content-Type: application/json' -d @$O/nm-req-$i.json > $O/nm-$n-$i.json
      log "lilicorr $n prompt $i: $(show $O/nm-$n-$i.json)"
    done
  else
    log "lilicorr $n: server did not come up: $(grep -E ' E ' $O/nm-$n.log | tail -2 | tr '\n' ' ' | cut -c1-240)"
  fi
  kill $S; wait $S 2>/dev/null; S=""; sleep 3
done
log "=== DRAFTERS DONE ==="
