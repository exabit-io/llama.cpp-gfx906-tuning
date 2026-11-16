#!/bin/bash
# mtp-depth.sh -- MTP decode at depth and the head-256 tile rule.
#
# Why: the package's rule (-mllvm -amdgpu-vgpr-threshold-percent=90 on fattn-tile-instance-dkq256-dv256, fork 290fc642a)
# was chosen on the prefill kernel <256,256,16,2> and accepted with llama-batched-bench, which decodes one token per
# stream per step (<1,2>). A per-object option changes every kernel of the object. codegen/tilescan.py (2026-11-16)
# shows that with threshold 90 the 8-column kernels (<4,2>, <2,4>, <1,8>) get 20 scratch reloads inside their loops
# where clang 23 and plain clang 24 have 1. <4,2> is the kernel a stream runs when it evaluates 3 or 4 tokens in one
# step, which is what MTP verification does (--spec-draft-n-max 2 or 3; settings/launch.sh uses 3; REQUIREMENTS R3.9
# puts MTP in scope for both profiles). Thresholds 88 and 85 leave no reload in any loop of any kernel of the object.
#
# Arms: one libggml-hip.so each, all linked from the ROCm 10.1 build's objects (/root/build-master-v101) with ONLY the
# tile object exchanged (codegen/drvexp.py, /root/swap/<arm>), preloaded under the installed package's llama-server and
# host libraries on the ROCm 10.1 runtime:
#   c23  the object from ROCm 10.0's clang 23        (in loops: <4,2> 1 reload, <16,2> 0)
#   c24  ROCm 10.1's clang 24, no rule               (<4,2> 1, <16,2> 10)
#   v90  clang 24 with threshold 90 = the package    (<4,2> 20, <16,2> 0)
#   v88  clang 24 with threshold 88                  (0 everywhere)
#   v85  clang 24 with threshold 85                  (0 everywhere)
# Cell = one server start: Qwen3.8-27B Q8_0, -sm tensor, f16 KV, 125 W, RCCL + gated custom AR, -np 1, -c 131072,
# --spec-type draft-mtp --spec-draft-n-max 3, --cache-ram 0. One request with a long prompt (the first CHARS characters
# of wikitext-2 test, about 115K tokens) that generates 512 tokens, then two more requests with the same text and
# another question (the slot keeps the text's KV, so only the question is evaluated), 512 tokens each. Greedy.
#   metric A, MTP decode at depth: the mean of the three requests' predicted_per_second
#   metric B, prefill: the long prompt's prompt_per_second (the <16,2> kernel; one sample per cell)
#
# READING, WRITTEN BEFORE THE FIRST CELL. n = N blocks (default 3), every arm once per block in a seeded random order.
# An arm is "down" ("up") against another on a metric when every one of its cells is below (above) every cell of the
# other; otherwise "level".
# (0) Validity: the generated text (SHA-256 per request) and the draft counts must be the same in every cell; a cell
#     that differs is discarded and reported. The session is read only if c24 is "down" against c23 on B (the known
#     prefill loss of the plain clang 24 object, 2.5 % at 1 x 128K).
# (1) Does the shipped rule cost MTP decode? Yes if v90 is "down" against c24 on A (same compiler, the kernel without
#     the 19 extra reloads). The size is the difference of the means.
# (2) A replacement threshold (v88, v85) qualifies if it is not "down" against c23 on B and not "down" against c24 on A.
# (3) The rule is changed only if (1) is yes with a difference of 0.5 % or more AND a replacement qualifies; the
#     replacement with the higher prefill mean is taken (88 on a tie within 0.1 %), and it then has to pass the two-cell
#     confirmation (gpu-opt-confirm.sh's rule) and the package acceptance before it ships. If (1) is no, or smaller
#     than 0.5 %, the rule stays and the static finding is recorded.
set -u
W=/root/rocm101-migration-20261114; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
F=/root/models/wikitext-2-raw/wiki.test.raw
N=${N:-3}; ARMS=${ARMS:-"c23 c24 v90 v88 v85"}; TAG=${TAG:-mtp-depth}; CHARS=${CHARS:-500000}; NPRED=${NPRED:-512}
P=$W/$TAG.progress; TSV=$W/$TAG.tsv; DONE=$W/.$TAG-done; O=$W/$TAG; mkdir -p $O
log(){ echo "$(date -Is) [mtpd] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE; echo $$ > $W/$TAG.pid
EXE=/opt/mx-llama.cpp/bin/llama-server; PORT=8098
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; touch $DONE; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; touch $DONE; exit 1; }
[ -s $TSV ] && { log "FATAL: $TSV exists"; touch $DONE; exit 1; }
dpkg-query -W -f '${Version}' mx-llama-cpp-gfx906 | grep -q 'rocm10\.1\.' || { log "FATAL: the installed package is not the ROCm 10.1 build"; touch $DONE; exit 1; }
lib(){ readlink -f /root/swap/$1/libggml-hip.so; }
for a in $ARMS; do
  l=$(lib $a); [ -f "$l" ] || { log "FATAL: arm $a has no library"; touch $DONE; exit 1; }
  got=$(LD_PRELOAD=$l ldd $EXE 2>/dev/null | awk '{ if ($2 == "=>") print $3; else print $1 }' | grep libggml-hip | xargs -r -n1 readlink -f | sort -u)
  [ "$got" = "$l" ] || { log "FATAL: arm $a would map $got"; touch $DONE; exit 1; }
  log "arm $a: $l ($(stat -c %s $l) bytes)"
done
python3 -I - "$F" "$O" "$CHARS" "$NPRED" <<'EOF'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()[:int(sys.argv[3])]
qs = ["Summarize the text above in one paragraph.", "List the five most important facts stated in the text above.",
      "Write a short critical review of the text above."]
for i, q in enumerate(qs, 1):
    json.dump({"prompt": text + "\n\n" + q + "\n", "n_predict": int(sys.argv[4]), "temperature": 0, "top_k": 1, "seed": 0,
               "cache_prompt": True}, open("%s/req-%d.json" % (sys.argv[2], i), "w"))
EOF
log "n=$N, arms: $ARMS; prompt $CHARS characters, $NPRED tokens per request; load $(cut -d' ' -f1-3 /proc/loadavg)"

. $R/gpu-test-env.sh
export GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
SPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${SPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/$TAG-clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/$TAG-smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $W/$TAG-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/$TAG-smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
log "$(state_line)"
field(){ # field FILE -> prompt_n prompt_tps predicted_n predicted_tps draft_accepted draft_n sha, tab separated; fails on an error reply
  python3 -I -c '
import hashlib, json, sys
d = json.load(open(sys.argv[1])); t = d["timings"]
print("\t".join(str(x) for x in (t["prompt_n"], "%.3f" % t["prompt_per_second"], t["predicted_n"], "%.4f" % t["predicted_per_second"],
      t.get("draft_n_accepted"), t.get("draft_n"), hashlib.sha256(d["content"].encode()).hexdigest()[:12])))' "$1" 2>/dev/null
}
cell(){ # cell ARM BLOCK
  local arm=$1 blk=$2 l got r f pre="" dec="" acc="" sha="" pn="" ok=1
  l=$(lib $arm)
  env LD_PRELOAD=$l AMD_COMGR_CACHE_DIR=/root/.cache/comgr-mtpdepth $EXE -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c 131072 -np 1 --cache-ram 0 --port $PORT \
    --spec-type draft-mtp --spec-draft-n-max 3 > $O/$arm-$blk-server.out 2> $O/$arm-$blk-server.log < /dev/null &
  SPID=$!
  if ! wait_server $PORT $SPID; then
    log "SERVER DID NOT COME UP: $arm b$blk: $(tail -2 $O/$arm-$blk-server.log | tr '\n' ' ' | cut -c1-200)"
    kpid $SPID; wait $SPID 2>/dev/null; SPID=""; printf "mtp\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$blk" >> $TSV; return 0
  fi
  got=$(awk '$6 ~ /libggml-hip/ {print $6}' /proc/$SPID/maps 2>/dev/null | sort -u | tr '\n' ' ')
  for r in 1 2 3; do
    curl -s --max-time 1800 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' -d @$O/req-$r.json > $O/$arm-$blk-$r.json
    if ! f=$(field $O/$arm-$blk-$r.json) || [ -z "$f" ]; then ok=0; log "REQUEST $r FAILED: $arm b$blk: $(head -c 200 $O/$arm-$blk-$r.json)"; break; fi
    [ $r = 1 ] && { pre=$(echo "$f" | cut -f2); pn=$(echo "$f" | cut -f1); }
    dec="$dec $(echo "$f" | cut -f4)"; acc="$acc $(echo "$f" | cut -f5)/$(echo "$f" | cut -f6)"; sha="$sha $(echo "$f" | cut -f7)"
    [ $r != 1 ] && [ "$(echo "$f" | cut -f1)" -gt 2000 ] && { ok=0; log "REQUEST $r re-evaluated $(echo "$f" | cut -f1) prompt tokens (the slot did not keep the text): $arm b$blk"; break; }
  done
  kpid $SPID; wait $SPID 2>/dev/null; SPID=""; sleep 3
  if [ "$got" != "$l " ]; then
    log "WRONG LIBRARY in $arm b$blk -- discarded: $got"; printf "mtp\t%s\t%s\tWRONGLIBS\tWRONGLIBS\n" "$arm" "$blk" >> $TSV; return 0
  fi
  if grep -qiE 'NCCL init failed|probe AllReduce failed' $O/$arm-$blk-server.log; then
    log "RCCL NOT UP in $arm b$blk -- discarded"; printf "mtp\t%s\t%s\tNORCCL\tNORCCL\n" "$arm" "$blk" >> $TSV; return 0
  fi
  if [ $ok = 0 ]; then printf "mtp\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$blk" >> $TSV; return 0; fi
  local mean
  mean=$(echo $dec | awk '{printf "%.4f", ($1 + $2 + $3) / 3}')
  printf "mtp\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$arm" "$blk" "$mean" "$pre" "$(echo $dec)" "$(echo $acc)" "$(echo $sha)" "$pn" >> $TSV
  log "$arm b$blk: MTP decode $mean ($(echo $dec)), prefill $pre over $pn tokens, accepted $(echo $acc), sha $(echo $sha)"
}
summary(){
  python3 -I - $TSV <<'EOF' 2>&1 | tee -a $P
import collections, statistics as st, sys
d = {}; p = {}; sig = collections.Counter(); rows = []
for l in open(sys.argv[1]):
    a = l.rstrip("\n").split("\t")
    try:
        x, y = float(a[3]), float(a[4])
    except ValueError:
        continue
    rows.append((a[1], a[2], x, y, (a[6], a[7]))); sig[(a[6], a[7])] += 1
if not rows: sys.exit()
good = sig.most_common(1)[0][0]
for arm, blk, x, y, s in rows:
    if s != good:
        print(f"[mtpd] DISCARDED {arm} b{blk}: text or draft counts differ from the majority"); continue
    d.setdefault(arm, []).append(x); p.setdefault(arm, []).append(y)
def rel(a, b): return "up" if min(a) > max(b) else "down" if max(a) < min(b) else "level"
for k in d:
    print(f"[mtpd] {k:4s} n={len(d[k])} MTP decode {st.mean(d[k]):.3f} ({min(d[k]):.3f}-{max(d[k]):.3f}); prefill {st.mean(p[k]):.1f} ({min(p[k]):.1f}-{max(p[k]):.1f})")
if "c23" in d and "c24" in d:
    print(f"[mtpd] control: c24 against c23 on prefill: {rel(p['c24'], p['c23'])} ({100 * (st.mean(p['c24']) / st.mean(p['c23']) - 1):+.2f} %)")
for k in d:
    if k in ("c23", "c24"): continue
    s = []
    if "c24" in d: s.append(f"MTP decode against c24 {rel(d[k], d['c24'])} ({100 * (st.mean(d[k]) / st.mean(d['c24']) - 1):+.2f} %)")
    if "c23" in d: s.append(f"against c23 {rel(d[k], d['c23'])} ({100 * (st.mean(d[k]) / st.mean(d['c23']) - 1):+.2f} %); prefill against c23 {rel(p[k], p['c23'])} ({100 * (st.mean(p[k]) / st.mean(p['c23']) - 1):+.2f} %)")
    print(f"[mtpd] {k}: " + ", ".join(s))
EOF
}
for blk in $(seq 1 $N); do
  order=$(python3 -I -c 'import random,sys; a=sys.argv[2:]; random.Random(int(sys.argv[1])).shuffle(a); print(" ".join(a))' "$(( 20261316 + blk ))" $ARMS)
  log "block $blk order: $order"
  for a in $order; do cell $a $blk; done
  log "=== block $blk DONE"
  summary
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore; touch $DONE; log "=== MTP-DEPTH DONE ==="
