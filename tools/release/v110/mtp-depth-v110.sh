#!/bin/bash
# mtp-depth-v110.sh -- MTP decode at depth and the head-256 tile rule on llama.cpp v1.1.0 (master-v110).
#
# Why: upstream #32901 changed how the tile kernel loads its data. Read statically (tile/tilescan-thresholds-v110.txt):
# the threshold 85 that -r2 ships now leaves 19 reloads in the loop of <8,1>; clang 23 itself has 18 in <4,2>, the
# kernel that verifies three drafted tokens (v1.0.0: 1); clang 24 without any rule has 13 there and none in <16,2>
# (prefill; v1.0.0: 16); thresholds 88 and 75 leave none in any of the 17 kernels. A count is not a timing result.
#
# Arms: one libggml-hip.so each, linked from the ROCm 10.1 build's objects (/root/build-master-v110-101) with ONLY the
# tile object exchanged (tile/tile-lib.sh), preloaded under that build's llama-server and host libraries, ROCm 10.1:
#   c23    the object from ROCm 10.0's clang 23           (in loops: <4,2> 18, <16,2> 0)
#   tnone  clang 24, no threshold                         (<4,2> 13, <16,2> 0)
#   t85    clang 24, threshold 85 = the rule as shipped   (<4,2> 0, <16,2> 0; <8,1> 19, not run here)
#   t88    clang 24, threshold 88                         (0 everywhere)
#   t75    clang 24, threshold 75                         (0 everywhere)
# Cell = one server start, as mtp-depth2.sh (2026-11-16): Qwen3.8-27B Q8_0, -sm tensor, f16 KV, 125 W, RCCL + gated
# custom AR, -np 1 (a ring context), -c 131072, --spec-type draft-mtp --spec-draft-n-max 3, --cache-ram 0; a long chat
# prompt (about 117.7K tokens) that generates NPRED tokens, then two more requests on the same text.
#   metric A, MTP decode at depth: the mean of the three requests' predicted_per_second
#   metric B, prefill: the long prompt's prompt_per_second (one sample per cell)
#
# READING, WRITTEN BEFORE THE FIRST CELL. n = N blocks (default 3), every arm once per block in a seeded random order.
# An arm is "down" ("up") against another on a metric when every one of its cells is below (above) every cell of the
# other; otherwise "level".
# (0) Validity: text (SHA-256 per request) and draft counts must be the same in every cell; a cell that differs loses
#     its decode sample and is reported.
# (1) Is a threshold still worth having? Yes if t88 or t75 is "up" against tnone on A by 0.5 % or more, or "up" on B.
#     If neither is, the rule is dropped for this kernel (tnone ships: nothing to maintain), provided tnone is not
#     "down" against c23 on either metric.
# (2) If (1) is yes: the threshold taken is the one of {88, 75} with the higher mean on A unless they tie within
#     0.25 %, then the higher mean on B, then 88 (nearer to what was shipped). It must not be "down" against c23 on B.
# (3) t85 is here to say what the shipped number does on the two kernels this cell runs; it is not a candidate
#     (not clean on <8,1>).
# (4) What is taken goes through the design-point cells (4 x 64K, 1 x 255K) against c23 and against the v1.0.0 -r2
#     build before a package is built; c23 "down" against the clang 24 arms on A is the size of what the ROCm 10.0
#     stream loses with this kernel and is reported.
set -u
W=/root/rebase-v110; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
F=/root/models/wikitext-2-raw/wiki.test.raw
N=${N:-3}; ARMS=${ARMS:-"c23 tnone t85 t88 t75"}; TAG=${TAG:-mtp-depth-v110}; CHARS=${CHARS:-500000}; NPRED=${NPRED:-768}
P=$W/$TAG.progress; TSV=$W/$TAG.tsv; DONE=$W/.$TAG-done; O=$W/$TAG; mkdir -p $O
log(){ echo "$(date -Is) [mtpd] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE; echo $$ > $W/$TAG.pid
BL=/root/build-master-v110-101; EXE=$BL/bin/llama-server; PORT=8098
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; touch $DONE; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; touch $DONE; exit 1; }
[ -s $TSV ] && { log "FATAL: $TSV exists"; touch $DONE; exit 1; }
lib(){ readlink -f /root/rebase-v110/tile/lib-$1/libggml-hip.so; }
for a in $ARMS; do
  l=$(lib $a); [ -f "$l" ] || { log "FATAL: arm $a has no library"; touch $DONE; exit 1; }
  got=$(LD_LIBRARY_PATH=$BL/bin LD_PRELOAD=$l ldd $EXE 2>/dev/null | awk '{ if ($2 == "=>") print $3; else print $1 }' | grep libggml-hip | xargs -r -n1 readlink -f | sort -u)
  [ "$got" = "$l" ] || { log "FATAL: arm $a would map $got"; touch $DONE; exit 1; }
  log "arm $a: $l ($(stat -c %s $l) bytes)"
done
python3 -I - "$F" "$O" "$CHARS" "$NPRED" <<'EOF'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()[:int(sys.argv[3])]
qs = ["Summarize every article in the text above in detail, one paragraph per article, in the order they appear.",
      "Go through the text above article by article and list every person named in it, with one sentence on who they are.",
      "Write a detailed critical review of each article in the text above, one paragraph per article, in the order they appear."]
for i, q in enumerate(qs, 1):
    json.dump({"messages": [{"role": "user", "content": text + "\n\n" + q}], "max_tokens": int(sys.argv[4]), "temperature": 0, "top_k": 1, "seed": 0,
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
d = json.load(open(sys.argv[1])); t = d["timings"]; m = d["choices"][0]["message"]
text = (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")
print("\t".join(str(x) for x in (t["prompt_n"], "%.3f" % t["prompt_per_second"], t["predicted_n"], "%.4f" % t["predicted_per_second"],
      t.get("draft_n_accepted"), t.get("draft_n"), hashlib.sha256(text.encode()).hexdigest()[:12])))' "$1" 2>/dev/null
}
cell(){ # cell ARM BLOCK
  local arm=$1 blk=$2 l got r f pre="" dec="" acc="" sha="" pn="" ok=1
  l=$(lib $arm)
  env LD_LIBRARY_PATH=$BL/bin LD_PRELOAD=$l AMD_COMGR_CACHE_DIR=/root/.cache/comgr-mtpdepth $EXE -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c 131072 -np 1 --cache-ram 0 --port $PORT \
    --spec-type draft-mtp --spec-draft-n-max 3 > $O/$arm-$blk-server.out 2> $O/$arm-$blk-server.log < /dev/null &
  SPID=$!
  if ! wait_server $PORT $SPID; then
    log "SERVER DID NOT COME UP: $arm b$blk: $(tail -2 $O/$arm-$blk-server.log | tr '\n' ' ' | cut -c1-200)"
    kpid $SPID; wait $SPID 2>/dev/null; SPID=""; printf "mtp\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$blk" >> $TSV; return 0
  fi
  got=$(awk '$6 ~ /libggml-hip/ {print $6}' /proc/$SPID/maps 2>/dev/null | sort -u | tr '\n' ' ')
  for r in 1 2 3; do
    curl -s --max-time 1800 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @$O/req-$r.json > $O/$arm-$blk-$r.json
    if ! f=$(field $O/$arm-$blk-$r.json) || [ -z "$f" ]; then ok=0; log "REQUEST $r FAILED: $arm b$blk: $(head -c 200 $O/$arm-$blk-$r.json)"; break; fi
    [ $r = 1 ] && { pre=$(echo "$f" | cut -f2); pn=$(echo "$f" | cut -f1); }
    dec="$dec $(echo "$f" | cut -f4)"; acc="$acc $(echo "$f" | cut -f5)/$(echo "$f" | cut -f6)"; sha="$sha $(echo "$f" | cut -f7)"
    [ "$(echo "$f" | cut -f3)" -ge $(( NPRED / 2 )) ] || { ok=0; log "REQUEST $r FAILED: $arm b$blk generated $(echo "$f" | cut -f3) tokens, fewer than half of $NPRED"; break; }
    [ "$(echo "$f" | cut -f3)" = "$NPRED" ] || log "note: $arm b$blk request $r ended by itself after $(echo "$f" | cut -f3) tokens"
    [ $r != 1 ] && [ "$(echo "$f" | cut -f1)" -gt 8192 ] && { ok=0; log "REQUEST $r FAILED: $arm b$blk re-evaluated $(echo "$f" | cut -f1) prompt tokens (the slot did not keep the text)"; break; }
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
    p.setdefault(arm, []).append(y)
    if s != good:
        print(f"[mtpd] DECODE DISCARDED {arm} b{blk}: text or draft counts differ from the majority (prefill kept)"); d.setdefault(arm, []); continue
    d.setdefault(arm, []).append(x)
def rel(a, b): return "up" if min(a) > max(b) else "down" if max(a) < min(b) else "level"
for k in d:
    print(f"[mtpd] {k:4s} MTP decode n={len(d[k])} " + (f"{st.mean(d[k]):.3f} ({min(d[k]):.3f}-{max(d[k]):.3f})" if d[k] else "none") + f"; prefill n={len(p[k])} {st.mean(p[k]):.1f} ({min(p[k]):.1f}-{max(p[k]):.1f})")
if "tnone" in d and "c23" in d and d["tnone"] and d["c23"]:
    print(f"[mtpd] tnone: MTP decode against c23 {rel(d['tnone'], d['c23'])} ({100 * (st.mean(d['tnone']) / st.mean(d['c23']) - 1):+.2f} %), prefill against c23 {rel(p['tnone'], p['c23'])} ({100 * (st.mean(p['tnone']) / st.mean(p['c23']) - 1):+.2f} %)")
for k in d:
    if k in ("c23", "tnone"): continue
    s = []
    if d[k] and d.get("tnone"): s.append(f"MTP decode against tnone {rel(d[k], d['tnone'])} ({100 * (st.mean(d[k]) / st.mean(d['tnone']) - 1):+.2f} %)")
    if d[k] and d.get("c23"): s.append(f"MTP decode against c23 {rel(d[k], d['c23'])} ({100 * (st.mean(d[k]) / st.mean(d['c23']) - 1):+.2f} %)")
    if "c23" in p: s.append(f"prefill against c23 {rel(p[k], p['c23'])} ({100 * (st.mean(p[k]) / st.mean(p['c23']) - 1):+.2f} %)")
    if "tnone" in p: s.append(f"prefill against tnone {rel(p[k], p['tnone'])} ({100 * (st.mean(p[k]) / st.mean(p['tnone']) - 1):+.2f} %)")
    print(f"[mtpd] {k}: " + ", ".join(s))
EOF
}
for blk in $(seq 1 $N); do
  order=$(python3 -I -c 'import random,sys; a=sys.argv[2:]; random.Random(int(sys.argv[1])).shuffle(a); print(" ".join(a))' "$(( 20261416 + blk ))" $ARMS)
  log "block $blk order: $order"
  for a in $order; do cell $a $blk; done
  log "=== block $blk DONE"
  summary
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore; touch $DONE; log "=== MTP-DEPTH DONE ==="
