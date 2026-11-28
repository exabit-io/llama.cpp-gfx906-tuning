#!/bin/bash
# firstreq-v110.sh [OUT] -- the slow start of a new llama-server process on the ROCm 10.1 runtime: how large it is,
# whether it is one-time or keeps coming back, and whether the runtime's deferred code loading carries it.
#
# Why: in the package test of 1.1.0 (deb-test-v110, 2026-11-27 18:36-18:53 UTC) the single-stream MTP requests of the
# installed package (ROCm 10.1 runtime) started slowly: the first request after the install 39.0 tok/s (101 tokens at
# 22 tok/s, then 75), the first request of the next two servers 55.5 and 58.2, two later requests 62.8 and 56.1 with
# the time lost in their last 30 tokens; steady 73.9-74.6. The same commit's source build on the ROCm 10.0 runtime, one
# hour earlier in the same queue: first requests 67.6-68.0, steady 74.1-74.9, no slow request in twelve. The v1.0.0
# cycle has it in its data too (pr-test-101: 54.8 on 10.1; 68.3 on 10.0). Nothing in the journal at those seconds, no
# clock latch. HIP loads the code of a translation unit on a device when one of its kernels is first launched
# (HIP_ENABLE_DEFERRED_LOADING, default on; the name is in both runtimes' libamdhip64): a candidate, not a finding.
#
# Arms, one server per cell: -sm tensor -np 1 -c 8192 --spec-type draft-mtp --spec-draft-n-max 2 (pr-test.sh's step
# 4), then REQ (12) times the same greedy 256-token request with cache_prompt false:
#   pkg    the installed package: its llama-server and libraries, ROCm 10.1 through DT_RPATH, no LD_LIBRARY_PATH
#   pkgE   the same with HIP_ENABLE_DEFERRED_LOADING=0
#   s101   /root/build-final-v110-101: the package's commit by ROCm 10.1 in another directory, ROCm 10.1 runtime
#   s100   /root/build-final-v110: the same commit by ROCm 10.0, ROCm 10.0 runtime
#   old    /root/build-v101-t85: v1.0.0 (the -r2 package's code) by ROCm 10.1, ROCm 10.1 runtime
# CHECK=1: only the arm checks (which libraries each arm would map), no GPU use.
# N blocks (default 3), every arm once per block in a seeded random order. Per request: seconds of generation, prompt
# milliseconds, text hash, and the growth of the server's evicted_ms (KFD: the time its GPU queues were evicted),
# summed over the four dies. Per server: seconds from exec to /health.
#
# READING, WRITTEN BEFORE THE FIRST RUN. Per cell: T1 = generation seconds of request 1; Ts = the median of requests
# 5..REQ; extra1 = T1 - Ts; a request from the 2nd on is LOST when it takes more than Ts + 0.3 s.
# "above" / "below": every block's value of one arm above / below every block's value of the other.
# (F1) One-time or recurring. An arm's slow start is ONE-TIME if no request from the 5th on is lost in any block;
#      otherwise it is RECURRING and the lost requests are listed with their index.
# (F2) Runtime and package. extra1 of s101 above s100's: the ROCm 10.1 stream starts slower than 10.0 (both medians
#      reported). pkg against s101 is the same code on the same runtime and is expected level; a median extra1 of pkg
#      more than 0.5 s over s101's with pkg above s101 means the PACKAGE adds a cost of its own: to the lead with the
#      package, before publishing.
# (F3) Deferred loading. extra1 of pkgE below pkg's with a median under half of pkg's: deferred loading carries the
#      start. Then pkgE's load time against pkg's is what the switch costs at start, and both go into the release notes
#      as a property of the runtime with the switch named. The package's default is NOT changed in this release (one
#      cell is not a test of the switch). Otherwise: not the carrier, the cause stays open and is recorded so.
# (F4) Release against release. Median extra1 of pkg more than 1 s over old's with pkg above old: named in the release
#      notes. pkg RECURRING and old not: to the lead before publishing.
# (F5) Evictions. A lost request (or a request 1) whose evicted_ms grew by more than 100 ms is reported as an eviction
#      of the GPU queues; without that growth it is not one.
# The rates of this script are one session's and go nowhere else. Texts: every request of a cell must give one hash
# (reported if not); arms may differ from each other (old is another release, s100 another compiler).
set -u
W=/root/rebase-v110; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
O=${1:-$W/firstreq-v110}; N=${N:-3}; REQ=${REQ:-12}; ARMS=${ARMS:-"pkg pkgE s101 s100 old"}; PORT=8097
PKG=/opt/mx-llama.cpp; S101=/root/build-final-v110-101; S100=/root/build-final-v110; OLD=/root/build-v101-t85
mkdir -p $O; P=$O/progress; DONE=$O/.done; T=$O/requests.tsv; rm -f $DONE
log(){ echo "$(date -Is) [firstreq] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
exe(){ case $1 in pkg|pkgE) echo $PKG/bin/llama-server;; s101) echo $S101/bin/llama-server;; s100) echo $S100/bin/llama-server;; old) echo $OLD/bin/llama-server;; esac; }
ldp(){ case $1 in pkg|pkgE) echo "";; s101) echo $S101/bin:/opt/rocm/core-10.1/lib;; s100) echo $S100/bin:/opt/rocm/core-10.0/lib;; old) echo $OLD/bin:/opt/rocm/core-10.1/lib;; esac; }
armenv(){ case $1 in pkgE) echo HIP_ENABLE_DEFERRED_LOADING=0;; *) echo FIRSTREQ_DUMMY=0;; esac; }
want(){ case $1 in pkg|pkgE) echo $PKG/lib;; s101) echo $S101/bin;; s100) echo $S100/bin;; old) echo $OLD/bin;; esac; }
rocm(){ case $1 in s100) echo core-10.0;; *) echo core-10.1;; esac; }
for a in $ARMS; do
  [ -x "$(exe $a)" ] || { log "FATAL: arm $a has no llama-server"; exit 1; }
  maps=$(env -u LD_LIBRARY_PATH LD_LIBRARY_PATH="$(ldp $a)" ldd $(exe $a) 2>/dev/null | awk '$2 == "=>" { print $3 }')
  g=$(echo "$maps" | grep libggml-hip | xargs -r -n1 readlink -f | xargs -r -n1 dirname | sort -u)
  h=$(echo "$maps" | grep libamdhip64 | xargs -r -n1 readlink -f | sort -u)
  [ "$g" = "$(readlink -f $(want $a))" ] || { log "FATAL: arm $a would map libggml-hip from $g"; exit 1; }
  case "$h" in */$(rocm $a)/*) ;; *) log "FATAL: arm $a would map $h, not $(rocm $a)"; exit 1;; esac
  log "arm $a: $(exe $a); libggml-hip in $g; $h"
done
[ "$(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906 2>/dev/null | cut -c1-5)" = "1.1.0" ] || { log "FATAL: the installed package is not 1.1.0"; exit 1; }
log "package $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906), source $(cut -c1-9 /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT 2>/dev/null); s101 $(cut -c1-9 $S101/.src-commit) s100 $(cut -c1-9 $S100/.src-commit); n=$N, $REQ requests per server; load $(cut -d' ' -f1-3 /proc/loadavg)"
[ "${CHECK:-0}" = 1 ] && { log "CHECK=1: arm checks only"; exit 0; }
for p in llama-server llama-batched-b llama-perplexit test-backend-op test-llama-cont llama-cli llama-completi cmake ninja; do
  pgrep -x $p >/dev/null && { log "FATAL: $p is running"; exit 1; }
done
busy=$(cat /sys/class/drm/card[0-9]/device/gpu_busy_percent 2>/dev/null | sort -n | tail -1)
[ "${busy:-0}" -lt 20 ] || { log "FATAL: a die is $busy % busy"; exit 1; }
echo '{"prompt":"Write a short paragraph about the history of the transistor.","n_predict":256,"temperature":0,"top_k":1,"seed":0,"cache_prompt":false}' > $O/req.json
. $R/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
SPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${SPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $O/clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$O/smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $O/watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $O/smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
log "$(state_line)"
evicted(){ local s=0 v f; for f in /sys/class/kfd/kfd/proc/$1/stats_*/evicted_ms; do [ -r "$f" ] && { v=$(cat "$f" 2>/dev/null); s=$(( s + ${v:-0} )); }; done; echo $s; }
now(){ date +%s.%N; }
: > $T
cell(){ # cell ARM BLOCK
  local arm=$1 blk=$2 t0 t1 up=0 i e0 e1 r f
  t0=$(now)
  if [ -n "$(ldp $arm)" ]; then
    env $(armenv $arm) LD_LIBRARY_PATH="$(ldp $arm)" $(exe $arm) -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -c 8192 -np 1 --port $PORT \
      --spec-type draft-mtp --spec-draft-n-max 2 > $O/$arm-$blk-server.out 2> $O/$arm-$blk-server.log < /dev/null &
  else
    env -u LD_LIBRARY_PATH $(armenv $arm) $(exe $arm) -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -c 8192 -np 1 --port $PORT \
      --spec-type draft-mtp --spec-draft-n-max 2 > $O/$arm-$blk-server.out 2> $O/$arm-$blk-server.log < /dev/null &
  fi
  SPID=$!
  for i in $(seq 1 900); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 0.5
  done
  t1=$(now)
  if [ $up != 1 ]; then log "$arm block $blk: server did not come up: $(tail -2 $O/$arm-$blk-server.log | tr '\n' ' ' | cut -c1-200)"; kpid $SPID; wait $SPID 2>/dev/null; SPID=""; sleep 3; return 1; fi
  for r in $(seq 1 $REQ); do
    e0=$(evicted $SPID)
    curl -s --max-time 600 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' -d @$O/req.json > $O/$arm-$blk-$r.json
    e1=$(evicted $SPID)
    f=$(python3 -I -c '
import hashlib, json, sys
d = json.load(open(sys.argv[1])); t = d["timings"]
print("\t".join(str(x) for x in ("%.4f" % (t["predicted_ms"] / 1000.0), "%.1f" % t["prompt_ms"], t["predicted_n"], t.get("draft_n_accepted"), t.get("draft_n"),
      hashlib.sha256(d.get("content", "").encode()).hexdigest()[:12])))' $O/$arm-$blk-$r.json 2>/dev/null) || f=""
    [ -n "$f" ] || { log "$arm block $blk request $r: no valid reply"; f=$(printf 'nan	nan	0	0	0	none'); }
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$blk" "$r" "$(python3 -I -c 'import sys; print("%.2f" % (float(sys.argv[2]) - float(sys.argv[1])))' $t0 $t1)" "$f" "$(( e1 - e0 ))" >> $T
  done
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; SPID=""; sleep 3
  log "$arm block $blk: load $(awk -F'\t' -v a=$arm -v b=$blk '$1 == a && $2 == b && $3 == 1 { print $4 }' $T) s; generation seconds per request: $(awk -F'\t' -v a=$arm -v b=$blk '$1 == a && $2 == b { printf "%.2f ", $5 }' $T); evicted ms: $(awk -F'\t' -v a=$arm -v b=$blk '$1 == a && $2 == b { printf "%s ", $11 }' $T)"
}
for blk in $(seq 1 $N); do
  order=$(python3 -I -c 'import random,sys; a=sys.argv[2:]; random.Random(int(sys.argv[1])).shuffle(a); print(" ".join(a))' "$(( 20261127 + blk ))" $ARMS)
  log "block $blk order: $order"
  for a in $order; do cell $a $blk; done
done
trap - INT TERM EXIT
kpid $SAMP; kpid $SMCL; touch $DONE; sleep 6; kpid $WDOG; restore
python3 -I - $T $REQ <<'EOF' | tee -a $P
import statistics, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]; REQ = int(sys.argv[2])
cells = {}
for arm, blk, r, load, sec, pms, n, acc, dn, sha, ev in rows:
    cells.setdefault(arm, {}).setdefault(int(blk), []).append((int(r), float(load), float(sec), float(pms), sha, int(ev), acc, dn))
res = {}
print("[firstreq] arm: per block load s | T1 s | steady s (median of requests 5..%d) | extra1 s | prompt ms of request 1 / steady | lost requests (index:seconds over) | evicted ms in request 1 | texts" % REQ)
for arm, blks in cells.items():
    ex, lost_late, loads = [], 0, []
    for blk in sorted(blks):
        c = sorted(blks[blk]); ts = statistics.median([x[2] for x in c if x[0] >= 5]); t1 = c[0][2]
        lost = [(x[0], x[2] - ts, x[5]) for x in c if x[0] >= 2 and x[2] > ts + 0.3]
        lost_late += sum(1 for x in lost if x[0] >= 5)
        ex.append(t1 - ts); loads.append(c[0][1])
        print("[firstreq] %-5s block %d: load %6.2f | T1 %.2f | steady %.3f | extra1 %+.2f | prompt %.0f / %.0f | lost %s | evicted %d | %s %s/%s" % (
            arm, blk, c[0][1], t1, ts, t1 - ts, c[0][3], statistics.median([x[3] for x in c if x[0] >= 5]),
            " ".join("%d:+%.2f(ev %d)" % x for x in lost) or "none", c[0][5], ",".join(sorted(set(x[4] for x in c))), c[0][6], c[0][7]))
    res[arm] = dict(ex=ex, late=lost_late, load=loads)
    print("[firstreq] %-5s extra1 median %+.2f s (%s); load median %.2f s; (F1) %s" % (arm, statistics.median(ex), " ".join("%+.2f" % v for v in ex),
          statistics.median(loads), "ONE-TIME" if lost_late == 0 else "RECURRING: %d lost requests from the 5th on" % lost_late))
def rel(a, b):
    if a not in res or b not in res: return None
    x, y = res[a]["ex"], res[b]["ex"]
    return "above" if min(x) > max(y) else "below" if max(x) < min(y) else "level"
def med(a): return statistics.median(res[a]["ex"])
if rel("s101", "s100"): print("[firstreq] (F2) s101 %s s100 (%+.2f against %+.2f s)" % (rel("s101", "s100"), med("s101"), med("s100")))
if rel("pkg", "s101"):
    own = rel("pkg", "s101") == "above" and med("pkg") - med("s101") > 0.5
    print("[firstreq] (F2) pkg %s s101 (%+.2f against %+.2f s): %s" % (rel("pkg", "s101"), med("pkg"), med("s101"), "THE PACKAGE ADDS A COST OF ITS OWN: to the lead before publishing" if own else "the package adds nothing of its own"))
if rel("pkgE", "pkg"):
    car = rel("pkgE", "pkg") == "below" and med("pkgE") < 0.5 * med("pkg")
    print("[firstreq] (F3) pkgE %s pkg (%+.2f against %+.2f s): %s; load %.2f against %.2f s" % (rel("pkgE", "pkg"), med("pkgE"), med("pkg"),
          "deferred loading carries the start" if car else "deferred loading is NOT shown as the carrier", statistics.median(res["pkgE"]["load"]), statistics.median(res["pkg"]["load"])))
if rel("pkg", "old"):
    named = rel("pkg", "old") == "above" and med("pkg") - med("old") > 1.0
    lead = res["pkg"]["late"] > 0 and res["old"]["late"] == 0
    print("[firstreq] (F4) pkg %s old (%+.2f against %+.2f s): %s%s" % (rel("pkg", "old"), med("pkg"), med("old"), "named in the release notes" if named else "no release-over-release cost by the rule",
          "; pkg RECURRING and old not: TO THE LEAD before publishing" if lead else ""))
EOF
log "=== FIRSTREQ DONE ==="
