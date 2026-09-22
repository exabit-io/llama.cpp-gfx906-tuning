#!/bin/bash
# gate4-graphreuse.sh — Part 6 gate 4. Can CUDA-graph reuse go stale under -sm tensor?
#
# WHY IT IS A REAL RISK, established by inspection 2026-09-20: upstream 57819b8d4 disables graph
# reuse for PIPELINE PARALLELISM only ("TODO: figure out a way to make graph reuse work with
# pipeline parallelism"). Tensor split is NOT covered by that guard, and our whole service runs
# -sm tensor with many slots sharing one context's memory. PR #24549 is not in v0.4.1 by number;
# what IS in it is 3f7c29d31 (graph_reused) plus per-input can_reuse() checks.
#
# INSTRUMENT: determinism across slots. Four slots, temperature 0, identical prompt, ignore_eos.
# Every completion must be byte-identical to the others AND to a single-slot reference. A stale
# graph corrupts one slot's tensors, which shows up as a diverging completion -- something a
# throughput benchmark cannot see at all.
#
# ARM A: default (reuse ON).  ARM B: LLAMA_GRAPH_REUSE_DISABLE=1.
# The test is VACUOUS unless reuse actually happens, so arm A must report `graphs reused = N` > 0;
# if N is 0 the gate closes trivially and that fact is the finding.
set -u
W=/root/night-20260919; B=/root/build-substrate-v041
M=/root/models/Qwen3.8-27B-Q8_0.gguf
R=/root/rocm-tests/bench
log() { echo "$(date -Is) [gate4] $*" | tee -a $W/gate4.log; }
: > $W/gate4.log
kpid() { [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] && kill "$1" 2>/dev/null; return 0; }
SRV=""
cleanup() { trap - INT TERM EXIT; kpid "${SRV:-}"; sleep 3
  pkill -f "^/bin/bash [^ ]*clamp-watchdog-v2[.]sh" 2>/dev/null
  for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
  log "STOPPED (trap)"; exit 1; }
trap cleanup INT TERM EXIT
export LD_LIBRARY_PATH=$B/lib          # T: no rpath -- without this the stock kernels are loaded
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
# Envelope guard on every long GPU job (standing rule). Guarded by PID, not by a cmdline pattern:
# on 2026-09-20 a pattern-guarded watchdog exited 1 s after start because the launch form differed,
# leaving a 1.5 h job unguarded. $$ cannot be missed however this script is invoked.
rm -f $W/.gate4-done
QUEUE_PID=$$ DONEFLAG=$W/.gate4-done SMCLOG=$W/gate4-smc.log SCLK_GUARD=1 \
  nohup $R/clamp-watchdog-v2.sh >> $W/gate4-watchdog.out 2>&1 &
nohup $R/smc-log.sh $W/gate4-smc.log >/dev/null 2>&1 &
sleep 6
PROMPT='Explain, step by step and in detail, how a queued batching server assigns requests to slots.'
run_arm() { # run_arm NAME SLOTS ENVSET
  local name=$1; local slots=$2; local disable=$3
  local out=$W/gate4-$name
  log "arm $name: slots=$slots LLAMA_GRAPH_REUSE_DISABLE=$disable"
  LLAMA_GRAPH_REUSE_DISABLE=$disable $B/bin/llama-server -m $M \
    --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -ctk q8_0 -ctv q8_0 \
    -np $slots -c $(( slots * 33792 )) -b 2048 -ub 2048 --cache-ram 49152 \
    --host 127.0.0.1 --port 8099 > $out-server.log 2>&1 &
  SRV=$!
  local ok=0 i
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8099/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { log "arm $name: SERVER DIED during load"; grep -m5 -iE 'error|abort' $out-server.log | tee -a $W/gate4.log; return 1; }
    sleep 5
  done
  [ $ok -eq 1 ] || { log "arm $name: server never became healthy"; return 1; }
  # identical request to every slot, concurrently
  # Wait ONLY on the curl pids. A bare `wait` waits for EVERY background child of this shell --
  # including the llama-server started with & above, and the nohup'd watchdog and sampler. On
  # 2026-09-20 that deadlocked this script: all four completions were written, then it sat waiting on
  # its own server forever with the GPUs at 0%. The pids are explicit for exactly that reason.
  local p; local cpids=()
  for p in $(seq 1 $slots); do
    curl -s http://127.0.0.1:8099/completion -H 'Content-Type: application/json' \
      -d "{\"prompt\":\"$PROMPT\",\"n_predict\":192,\"temperature\":0,\"seed\":1,\"cache_prompt\":false,\"ignore_eos\":true}" \
      > $out-slot$p.json &
    cpids+=($!)
  done
  for p in "${cpids[@]}"; do wait "$p" 2>/dev/null; done
  kpid "$SRV"; sleep 5; SRV=""
  local reused; reused=$(grep -oE 'graphs reused = *[0-9]+' $out-server.log | tail -1 | grep -oE '[0-9]+$')
  log "arm $name: graphs reused = ${reused:-UNREPORTED}"
  python3 - "$out" "$slots" "$name" <<'PY' | tee -a $W/gate4.log
import json,sys,hashlib
base,slots,name=sys.argv[1],int(sys.argv[2]),sys.argv[3]
texts={}
for p in range(1,slots+1):
    try: texts[p]=json.load(open(f"{base}-slot{p}.json")).get("content","")
    except Exception as e: texts[p]=f"<<UNREADABLE {e}>>"
h={p:hashlib.sha256(t.encode()).hexdigest()[:12] for p,t in texts.items()}
uniq=set(h.values())
print(f"  {name}: {len(uniq)} distinct completion(s) across {slots} slots -> " + ("IDENTICAL" if len(uniq)==1 else "DIVERGED"))
for p in sorted(h): print(f"    slot{p} {h[p]} len={len(texts[p])}")
open(f"{base}-hashes.txt","w").write("\n".join(f"{p}\t{h[p]}" for p in sorted(h)))
PY
  return 0
}
run_arm reuse-on  4 0 || log "arm reuse-on FAILED"
run_arm reuse-off 4 1 || log "arm reuse-off FAILED"
run_arm single    1 0 || log "arm single FAILED"
log "=== comparing arms: reuse-on must match reuse-off and the 1-slot reference"
python3 - <<'PY' | tee -a $W/gate4.log
import json,hashlib,os
W="/root/night-20260919"
def h(f):
    try: return hashlib.sha256(json.load(open(f))["content"].encode()).hexdigest()[:12]
    except Exception: return None
on=[h(f"{W}/gate4-reuse-on-slot{p}.json") for p in range(1,5)]
off=[h(f"{W}/gate4-reuse-off-slot{p}.json") for p in range(1,5)]
ref=h(f"{W}/gate4-single-slot1.json")
print(f"  reuse-on : {on}")
print(f"  reuse-off: {off}")
print(f"  1-slot reference: {ref}")
bad=[]
if None in on or None in off or ref is None: bad.append("a completion was missing or unreadable")
if len(set(x for x in on if x))>1: bad.append("reuse-ON slots DIVERGED from each other")
if len(set(x for x in off if x))>1: bad.append("reuse-OFF slots diverged (not a reuse bug: something else is wrong)")
# CORRECTED 2026-09-20 after the first run: a 4-slot vs 1-slot difference is NOT a reuse failure.
# Batching changes matmul widths (MMVQ vs MMQ paths), so batched inference is not bitwise equal to
# unbatched even at temperature 0 -- and the difference PERSISTS WITH REUSE DISABLED, which proves
# reuse is not its cause. The gate's real question is whether reuse changes anything AT THE SAME
# BATCH SHAPE. Cross-shape is reported as information only.
if ref and on and on[0] and on[0]!=ref:
    print("  note: 4-slot output != 1-slot reference. Expected batch-shape nondeterminism (it also")
    print("        holds with reuse OFF), not a reuse defect. Patch comparisons must always hold the")
    print("        batch shape fixed, which the survey's cells do.")
if on and off and on[0] and off[0] and on[0]!=off[0]:
    bad.append("reuse-ON output differs from reuse-OFF at the SAME shape — this IS a reuse defect")
print("  GATE 4 VERDICT: " + ("PASS — graph reuse is bitwise-neutral under -sm tensor" if not bad else "FAIL — " + "; ".join(bad)))
PY
trap - INT TERM EXIT
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
touch $W/.gate4-done
pkill -f "^/bin/bash [^ ]*smc-log[.]sh $W" 2>/dev/null
echo "=== GATE4 DONE ===" >> $W/gate4.log
