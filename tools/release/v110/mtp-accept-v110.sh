#!/bin/bash
# mtp-accept-v110.sh -- the MTP-at-depth part of the acceptance of the 1.1.0 package (mtp-depth-v110.sh's cell and code;
# only the arms, the header and one summary line differ).
#
# Why: the package's GPU code changes in the head-256 tile object (upstream #32901 rewrote the kernel; the LLVM 24 rule
# is 88 now, master 1c52068f8), and llama-batched-bench does not run the kernel that verifies drafted tokens. The
# threshold was chosen on a library made by exchanging one object (mtp-depth-v110b, 2026-11-27: t88 +7.2 % MTP decode
# over the clang 23 object at 118K tokens of depth). This reads the INSTALLED PACKAGE with the same cell.
# Arms (-np 1 -c 131072, --spec-type draft-mtp --spec-draft-n-max 3, a 117.7K-token chat prompt, three requests of 768):
#   pkg   the installed package: its llama-server, its host libraries, its libggml-hip.so, ROCm 10.1 through DT_RPATH.
#         Its all-reduce gate is the built-in default (GGML_TP_AR_MAX_NE empty); at one slot every message is 1 to 4
#         rows wide and takes the custom path under the old and the new default alike.
#   t88   /root/build-master-v110-101 (master 9a7177d0a by ROCm 10.1) with tile/lib-t88: the library the threshold was
#         chosen on
#   c23   the same build with the tile object from ROCm 10.0's clang 23 (what the ROCm 10.0 stream's kernel does)
#   old   v1.0.0 master (224b73e24 = the -r2 package's code) by ROCm 10.1, /root/build-v101-t85: another release's
#         text, so its decode is shown and not compared by rule; its prefill is comparable.
#
# READING, WRITTEN BEFORE THE FIRST CELL. n = N blocks (default 3), every arm once per block in a seeded random order.
# "up" / "down": every sample of one arm above / below every sample of the other; otherwise "level".
# (0) Decode is compared only between cells with the same generated text and draft counts. pkg, t88 and c23 are
#     expected to write the same text (the package's three commits over 9a7177d0a do not act at one slot). If pkg writes
#     another text, its decode has no partner: that is reported, its prefill is still read, and the package's MTP row
#     paths are covered by deb-test.sh (mtp-modes.sh) instead.
# (1) The package passes this part if its MTP decode is "up" against c23 with an effect of at least +3.5 % (t88 read
#     +7.2 % when the threshold was chosen; half of that is the bar, as for -r2) AND it is not "down" against t88 with
#     an effect below -1 % AND its prefill is not "down" against t88 with an effect below -0.5 %.
# (2) pkg against old, prefill: more than 2 % under old is a cost of the release at depth and goes to the lead with the
#     package (rule (5) of mtp-depth-v110.sh; t88 read -0.11 %).
# (3) If (1) fails the package is not published; the difference between pkg and t88 is then located by swapping
#     libraries (pkg-swap.sh), not by rebuilding.
# WANT=<commit> in the environment: the source commit the installed package must be built from.
set -u
W=/root/rebase-v110; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
F=/root/models/wikitext-2-raw/wiki.test.raw
N=${N:-3}; ARMS=${ARMS:-"pkg t88 c23 old"}; TAG=${TAG:-mtp-accept-v110}; CHARS=${CHARS:-500000}; NPRED=${NPRED:-768}
P=$W/$TAG.progress; TSV=$W/$TAG.tsv; DONE=$W/.$TAG-done; O=$W/$TAG; mkdir -p $O
log(){ echo "$(date -Is) [mtpd] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE; echo $$ > $W/$TAG.pid
BL=/root/build-master-v110-101; EXE=$BL/bin/llama-server; PORT=8098
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; touch $DONE; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; touch $DONE; exit 1; }
[ -s $TSV ] && { log "FATAL: $TSV exists"; touch $DONE; exit 1; }
OLDB=/root/build-v101-t85
PKGD=/opt/mx-llama.cpp
lib(){ case $1 in old) readlink -f $OLDB/bin/libggml-hip.so;; pkg) readlink -f $PKGD/lib/libggml-hip.so;; *) readlink -f /root/rebase-v110/tile/lib-$1/libggml-hip.so;; esac; }
exe(){ case $1 in old) echo $OLDB/bin/llama-server;; pkg) echo $PKGD/bin/llama-server;; *) echo $EXE;; esac; }
libdir(){ case $1 in old) echo $OLDB/bin;; pkg) echo $PKGD/lib;; *) echo $BL/bin;; esac; }
# the package runs its own built-in all-reduce gate (an empty value is "unset" to the code); the other arms keep 20481
armenv(){ case $1 in pkg) echo GGML_TP_AR_MAX_NE=;; *) echo GGML_TP_AR_MAX_NE=20481;; esac; }
V=$(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906 2>/dev/null); SC=$(cat /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT 2>/dev/null)
case " $ARMS " in *" pkg "*) case $V in 1.1.0*.rocm10.1.gfx906.*) ;; *) log "FATAL: the installed package $V is not a v1.1.0 ROCm 10.1 build"; touch $DONE; exit 1;; esac
  [ -z "${WANT:-}" ] || [ "${SC:0:9}" = "${WANT:0:9}" ] || { log "FATAL: the installed package is built from ${SC:0:9}, not $WANT"; touch $DONE; exit 1; }
  log "package $V, source ${SC:0:9}; tile rule in the package build: $(grep -oE 'amdgpu-vgpr-threshold-percent=[0-9]+' /root/mx-llama-deb/build/compile_commands.json 2>/dev/null | sort | uniq -c | tr '\n' ' ')";; esac
for a in $ARMS; do
  l=$(lib $a); [ -f "$l" ] || { log "FATAL: arm $a has no library"; touch $DONE; exit 1; }
  got=$(LD_LIBRARY_PATH=$(libdir $a) LD_PRELOAD=$l ldd $(exe $a) 2>/dev/null | awk '{ if ($2 == "=>") print $3; else print $1 }' | grep libggml-hip | xargs -r -n1 readlink -f | sort -u)
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
  env $(armenv $arm) LD_LIBRARY_PATH=$(libdir $arm) LD_PRELOAD=$l AMD_COMGR_CACHE_DIR=/root/.cache/comgr-mtpdepth $(exe $arm) -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
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
    rows.append((a[1], a[2], x, y, (a[6], a[7])))
    if a[1] != "old": sig[(a[6], a[7])] += 1
if not rows or not sig: sys.exit()
good = sig.most_common(1)[0][0]
oldd = []
for arm, blk, x, y, s in rows:
    p.setdefault(arm, []).append(y)
    if arm == "old":
        oldd.append((x, s[0])); continue
    if s != good:
        print(f"[mtpd] DECODE DISCARDED {arm} b{blk}: text or draft counts differ from the majority (prefill kept)"); d.setdefault(arm, []); continue
    d.setdefault(arm, []).append(x)
def rel(a, b): return "up" if min(a) > max(b) else "down" if max(a) < min(b) else "level"
for k in d:
    print(f"[mtpd] {k:4s} MTP decode n={len(d[k])} " + (f"{st.mean(d[k]):.3f} ({min(d[k]):.3f}-{max(d[k]):.3f})" if d[k] else "none") + f"; prefill n={len(p[k])} {st.mean(p[k]):.1f} ({min(p[k]):.1f}-{max(p[k]):.1f})")
if oldd:
    print(f"[mtpd] old  (v1.0.0, another text; decode NOT compared by rule) MTP decode n={len(oldd)} {st.mean([x for x, _ in oldd]):.3f} ({min(x for x, _ in oldd):.3f}-{max(x for x, _ in oldd):.3f}), accepted {sorted(set(a for _, a in oldd))}; prefill n={len(p['old'])} {st.mean(p['old']):.1f} ({min(p['old']):.1f}-{max(p['old']):.1f})")
    print(f"[mtpd] v1.1.0 accepted/drafted in the tile arms: {good[0]}")
    for k in d:
        if k in p and p.get("old"): print(f"[mtpd] {k}: prefill against old {rel(p[k], p['old'])} ({100 * (st.mean(p[k]) / st.mean(p['old']) - 1):+.2f} %)" + (f", MTP decode ratio to old {st.mean(d[k]) / st.mean([x for x, _ in oldd]):.4f} (other text)" if d[k] else ""))
if "tnone" in d and "c23" in d and d["tnone"] and d["c23"]:
    print(f"[mtpd] tnone: MTP decode against c23 {rel(d['tnone'], d['c23'])} ({100 * (st.mean(d['tnone']) / st.mean(d['c23']) - 1):+.2f} %), prefill against c23 {rel(p['tnone'], p['c23'])} ({100 * (st.mean(p['tnone']) / st.mean(p['c23']) - 1):+.2f} %)")
if d.get("pkg") and d.get("t88"):
    print(f"[mtpd] pkg: MTP decode against t88 {rel(d['pkg'], d['t88'])} ({100 * (st.mean(d['pkg']) / st.mean(d['t88']) - 1):+.2f} %), prefill against t88 {rel(p['pkg'], p['t88'])} ({100 * (st.mean(p['pkg']) / st.mean(p['t88']) - 1):+.2f} %)")
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
restore; touch $DONE; log "=== MTP-ACCEPT-V110 DONE ==="
