#!/bin/bash
# round2-smoke.sh -- second round of functional tests, on the abliterated models, with the installed deb (/opt/mx-llama.cpp).
# SMOKE ONLY: these depths are below the 64K floor, so no number here is a result (REQUIREMENTS R2.2).
set -u
B=/opt/mx-llama.cpp; R=/root/rocm-tests/bench; MD=/root/models; O=/root/rebase-v070/round2; mkdir -p $O
F=$MD/wikitext-2-raw/wiki.test.raw
FG=$MD/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF
declare -A M27=(
  [huihui-q8]=$MD/Huihui-Qwen3.8-27B-abliterated-Q8_0.gguf [huihui-q6]=$MD/Huihui-Qwen3.8-27B-abliterated-Q6_K.gguf
  [lam12-q8]=$MD/Qwen3.8-27B-abliterated-mtp-Q8_0.gguf  [lam12-q6]=$MD/Qwen3.8-27B-abliterated-mtp-Q6_K.gguf )
D4="--device rocm0,rocm1,rocm2,rocm3"
P=$O/progress; DONE=$O/.done; rm -f $DONE; echo $$ > $O/pid
log(){ echo "$(date -Is) [r2] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
. $R/gpu-test-env.sh
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml HSA_FORCE_FINE_GRAIN_PCIE=1 LLAMA_PLE_SHARD=1
BPID=""; SPID=""
cleanup(){ trap - INT TERM EXIT; for p in "${BPID:-}" "${SPID:-}" "${WDOG:-}" "${SMCL:-}"; do kpid "$p"; done; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$O/smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $O/watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $O/smc.log >/dev/null 2>&1 &
SMCL=$!
run(){ local name=$1; shift; "$@" > $O/$name.out 2> $O/$name.log < /dev/null & BPID=$!; wait $BPID; local rc=$?; BPID=""; return $rc; }
mtp(){ # mtp NAME PORT ARGS... -- llama-server with draft-mtp n_max 2, one 256-token greedy completion
  local name=$1 port=$2; shift 2
  $B/bin/llama-server "$@" -fa on -np 1 --port $port --spec-type draft-mtp --spec-draft-n-max 2 > $O/$name.out 2> $O/$name.log < /dev/null &
  SPID=$!
  if wait_server $port $SPID; then
    curl -s --max-time 900 http://127.0.0.1:$port/completion -H 'Content-Type: application/json' \
      -d '{"prompt":"Write a short paragraph about the history of the transistor.","n_predict":256,"temperature":0,"top_k":1,"seed":0}' > $O/$name.json
    log "  $name: $(python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); t=d["timings"]; print("tokens", t["predicted_n"], "tok/s %.1f" % t["predicted_per_second"], "draft", t.get("draft_n"), "accepted", t.get("draft_n_accepted"), "| text:", d["content"][:90].replace(chr(10)," "))' $O/$name.json 2>&1)"
  else
    log "  $name: server did not come up: $(grep -iE 'error|fail' $O/$name.log | tail -2 | tr '\n' ' ' | cut -c1-300)"
  fi
  kpid "$SPID"; wait "$SPID" 2>/dev/null; SPID=""; sleep 3
}
log "round 2 smoke on $B ($(cat /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT | cut -c1-9)); $(state_line)"

log "1. Qwen3.8-27B abliterated: perplexity 16K x 6 (smoke), -sm layer -lm dio greedy, MTP"
port=8093
for k in huihui-q8 huihui-q6 lam12-q8 lam12-q6; do
  m=${M27[$k]}
  run ppl-$k timeout 3600 $B/bin/llama-perplexity -m $m $D4 -sm tensor -ngl all -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F
  log "  $k ppl rc=$? $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $O/ppl-$k.log | tail -1)"
  run gen-$k timeout 900 $B/bin/llama-completion -m $m $D4 -sm layer -ngl all -fa on -lm dio -c 4096 -n 64 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup -p "Explain in two sentences why the sky is blue."
  log "  $k gen rc=$? dio-fallbacks=$(grep -c 'Falling back' $O/gen-$k.log): $(tr '\n' ' ' < $O/gen-$k.out | cut -c1-160)"
  mtp mtp-$k $port -m $m $D4 -sm tensor -ngl all -c 8192
  port=$((port+1))
done

log "2. vision: mmproj on Huihui Q8_0 reading a generated image"
python3 -I -c '
from PIL import Image, ImageDraw, ImageFont
im = Image.new("RGB", (640, 360), "white"); d = ImageDraw.Draw(im)
d.ellipse((40, 90, 220, 270), fill="red")
try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 54)
except Exception: f = ImageFont.load_default()
d.text((260, 150), "EXABIT 2026", fill="black", font=f)
im.save("/root/rebase-v070/round2/test-image.png")'
run vision-huihui timeout 900 $B/bin/llama-mtmd-cli -m ${M27[huihui-q8]} --mmproj $MD/Huihui-Qwen3.8-27B-abliterated-mmproj-bf16.gguf $D4 -sm layer -ngl all -fa on -c 8192 -n 400 --temp 0 --top-k 1 -s 0 \
  --image $O/test-image.png -p "What text is written in this image, and what shape and colour is the object on the left? Answer briefly."
log "  vision rc=$?: $(tr '\n' ' ' < $O/vision-huihui.out | tail -c 300)"

log "3. Qwen3.8-Flash-Next abliterated (qwen4exp)"
FN4=$FG/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
FN8=$FG/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
run fn4-gen timeout 1800 $B/bin/llama-completion -m $FN4 $D4 -sm tensor -ngl all -lm mlock --n-cpu-moe 41 -fa on -c 4096 -n 64 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup -p "Explain in two sentences why the sky is blue."
log "  UD-Q4_K_XL gen rc=$?: $(tr '\n' ' ' < $O/fn4-gen.out | cut -c1-160)"
mtp fn4-mtp 8099 -m $FN4 -md $FG/mtp-Huihui-Qwen3.8-Flash-Next-abliterated-Q8_0.gguf $D4 -sm tensor -ngl all -lm mlock --n-cpu-moe 41 -c 8192
run fn8-gen timeout 2400 $B/bin/llama-completion -m $FN8 $D4 -sm tensor -ngl all -lm mlock --n-cpu-moe 99 -fa on -c 4096 -n 64 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup -p "Explain in two sentences why the sky is blue."
log "  Q8_0 gen rc=$?: $(tr '\n' ' ' < $O/fn8-gen.out | cut -c1-160) | $(grep -iE 'out of memory|failed to|error' $O/fn8-gen.log | head -2 | tr '\n' ' ' | cut -c1-200)"
run fn4-vision timeout 1800 $B/bin/llama-mtmd-cli -m $FN4 --mmproj $FG/mmproj-Qwen3.8-Flash-Next-BF16.gguf $D4 -sm tensor -ngl all -lm mlock --n-cpu-moe 41 -fa on -c 8192 -n 400 --temp 0 --top-k 1 -s 0 \
  --image $O/test-image.png -p "What text is written in this image, and what shape and colour is the object on the left? Answer briefly."
log "  Flash-Next vision rc=$?: $(tr '\n' ' ' < $O/fn4-vision.out | tail -c 300)"

trap - INT TERM EXIT
kpid "${WDOG:-}"; kpid "${SMCL:-}"; restore; touch $DONE; log "=== ROUND 2 SMOKE DONE ==="
