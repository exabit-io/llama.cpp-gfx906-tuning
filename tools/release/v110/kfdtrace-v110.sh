#!/bin/bash
# kfdtrace-v110.sh [OUT] -- which kernel path evicts a llama-server's GPU queues, and what calls it.
#
# Why: firstreq-v110.sh (2026-11-28) found evicted_ms of the server growing by about 150 ms (sum over four dies) in
# every request from the third on, in every arm (v1.0.0 too), sometimes stopping after the fifth; and every
# llama-batched-bench process of c3-rerun has one eviction of about 700 ms during the model load. The pauses of the
# acceptance evening (5 to 20 s, all dies idle) have no known cause; queue eviction is the first candidate.
#
# What: one server of the installed package (firstreq's command line), 8 requests. From before the start to the end
# the kernel's function tracer records, with the call stack, every entry into the three ways KFD evicts a process:
#   amdgpu_amdkfd_evict_userptr                  a host range registered with the GPU was invalidated (MMU notifier)
#   svm_range_evict                              the same for a shared-virtual-memory range
#   kgd2kfd_schedule_evict_and_restore_process   a buffer of the process is being moved (eviction fence)
# and kgd2kfd_quiesce_mm / kgd2kfd_resume_mm (the stop and the restart of the queues). Only these five functions are
# traced: no measurable load. The tracer is put back as it was (nop, tracing_on 1) on every exit.
#
# READING, WRITTEN BEFORE THE RUN. No rule for the package (it is published; the evictions are in v1.0.0 too).
#   (K1) The script prints, per traced function and distinct stack, the count and the stack. The path with entries at
#        the times evicted_ms grows is the one that evicts; the bottom of its stack names the trigger (a system call
#        of the server = its own memory handling; a kernel thread = the kernel's).
#   (K2) If none of the five functions is entered while evicted_ms grows, the counter is fed by another path and the
#        trace list must be widened (kfd_process_evict_queues with stack).
# The result is a located cause or a narrower question for the open item "pauses"; it says nothing about the pauses
# themselves unless one happens in this minute.
set -u
W=/root/rebase-v110; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
O=${1:-$W/kfdtrace-v110}; PORT=8097; REQ=${REQ:-8}; PKG=/opt/mx-llama.cpp; TR=/sys/kernel/tracing
mkdir -p $O; P=$O/progress; DONE=$O/.done; rm -f $DONE
log(){ echo "$(date -Is) [kfdtrace] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
for p in llama-server llama-batched-b llama-perplexit test-backend-op llama-cli cmake ninja; do
  pgrep -x $p >/dev/null && { log "FATAL: $p is running"; exit 1; }
done
[ "$(cat $TR/current_tracer)" = nop ] || { log "FATAL: a tracer is active: $(cat $TR/current_tracer)"; exit 1; }
FUNCS="amdgpu_amdkfd_evict_userptr svm_range_evict.isra.0 kgd2kfd_schedule_evict_and_restore_process kgd2kfd_quiesce_mm kgd2kfd_resume_mm"
for f in $FUNCS; do grep -q "^$f " $TR/available_filter_functions || { log "FATAL: $f cannot be traced"; exit 1; }; done
troff(){ echo 0 > $TR/tracing_on; echo nop > $TR/current_tracer; echo 0 > $TR/options/func_stack_trace; echo > $TR/set_ftrace_filter; echo 1 > $TR/tracing_on; }
. $R/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
SPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${SPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; touch $O/ev.log.stop; troff; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $O/clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$O/smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $O/watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $O/smc.log >/dev/null 2>&1 &
SMCL=$!
bash $W/evsample.sh $O/ev.log $O/ev.log.stop & EVS=$!
sleep 6
log "$(state_line)"
echo 0 > $TR/tracing_on; echo > $TR/trace
echo "$FUNCS" > $TR/set_ftrace_filter
echo 1 > $TR/options/func_stack_trace
echo function > $TR/current_tracer
echo 1 > $TR/tracing_on
log "tracer on: $(cat $TR/set_ftrace_filter | tr '\n' ' ')"
echo '{"prompt":"Write a short paragraph about the history of the transistor.","n_predict":256,"temperature":0,"top_k":1,"seed":0,"cache_prompt":false}' > $O/req.json
env -u LD_LIBRARY_PATH $PKG/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -c 8192 -np 1 --port $PORT \
  --spec-type draft-mtp --spec-draft-n-max 2 > $O/server.out 2> $O/server.log < /dev/null &
SPID=$!
up=0
for i in $(seq 1 900); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }
  kill -0 $SPID 2>/dev/null || break
  sleep 0.5
done
[ $up = 1 ] || { log "FATAL: the server did not come up"; cleanup; }
log "server pid $SPID up; uptime clock $(cut -d' ' -f1 /proc/uptime)"
evicted(){ local s=0 v f; for f in /sys/class/kfd/kfd/proc/$1/stats_*/evicted_ms; do [ -r "$f" ] && { v=$(cat "$f" 2>/dev/null); s=$(( s + ${v:-0} )); }; done; echo $s; }
for r in $(seq 1 $REQ); do
  e0=$(evicted $SPID); u0=$(cut -d' ' -f1 /proc/uptime)
  curl -s --max-time 600 http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' -d @$O/req.json > $O/req-$r.json
  e1=$(evicted $SPID); u1=$(cut -d' ' -f1 /proc/uptime)
  log "request $r: uptime clock $u0 to $u1, evicted_ms +$(( e1 - e0 )), $(python3 -I -c 'import json,sys; t=json.load(open(sys.argv[1]))["timings"]; print("%.2f s generation, prompt %.0f ms" % (t["predicted_ms"]/1000, t["prompt_ms"]))' $O/req-$r.json 2>/dev/null)"
done
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; SPID=""; sleep 2
echo 0 > $TR/tracing_on; cat $TR/trace > $O/trace.txt; troff
touch $O/ev.log.stop; wait $EVS 2>/dev/null
trap - INT TERM EXIT
kpid $SAMP; kpid $SMCL; touch $DONE; sleep 6; kpid $WDOG; restore
log "trace: $(grep -c . $O/trace.txt) lines, $(grep -c ' <-' $O/trace.txt) function entries"
python3 -I - $O/trace.txt <<'PY' | tee -a $P
import collections, re, sys
ev = []; cur = None
for l in open(sys.argv[1]):
    m = re.match(r'\s*(.+?)-(\d+)\s+\[(\d+)\]\s+\S+\s+([\d.]+):\s+(\S+) <-(\S+)', l)
    if m:
        cur = {'comm': m.group(1).strip(), 'pid': m.group(2), 't': float(m.group(4)), 'f': m.group(5), 'stack': []}; ev.append(cur); continue
    m = re.match(r'\s*=> (\S+)', l)
    if m and cur is not None: cur['stack'].append(m.group(1))
g = collections.defaultdict(list)
for e in ev: g[(e['f'], tuple(e['stack'][:22]))].append(e)
print('[kfdtrace] %d entries, %d distinct (function, stack)' % (len(ev), len(g)))
for (f, st), v in sorted(g.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
    comms = collections.Counter(e['comm'] for e in v)
    print('[kfdtrace] %s: %d entries, uptime clock %.2f to %.2f, by %s' % (f, len(v), v[0]['t'], v[-1]['t'], dict(comms)))
    print('[kfdtrace]    ' + ' <- '.join(st[1:]))
PY
log "=== KFDTRACE DONE ==="
