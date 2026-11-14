#!/bin/bash
# stall-diag.sh OUT [BUILD] [NREQ] -- where do the 1-2 s stalls of the -np 4 single-stream MTP requests sit?
# np4-ab.sh (2026-11-14) saw requests on BOTH masters (v0.9.0, v1.0.0) drop from 76-78 to 46-67 tok/s, the time lost in
# the last tokens of a request or in its prompt, with identical graph-reallocation counts. This runs the same requests
# against one server with a verbose log (every llama/ggml line is timestamped) and reports, per request, the speed and
# the largest gaps between consecutive log lines with the lines around them. Before and after: SMI count (MSR 0x34),
# memory CE count, VRAM in use per die, CPU pressure. Waits for drafters-100.sh (the GPUs are its until then).
set -u
W=/root/rebase-v100; O=$1; B=${2:-/opt/mx-llama.cpp}; NREQ=${3:-8}
mkdir -p $O; P=$O/progress
M=/root/models/Qwen3.8-27B-Q8_0.gguf
log(){ echo "$(date -Is) [stall] $*" | tee -a $P; }
until grep -q 'DRAFTERS DONE' $W/drafters/progress 2>/dev/null; do sleep 15; done
[ -f $W/np4-ab/v100-host-1/req-short.json ] || { log "no request files"; exit 1; }
. /root/rocm-tests/bench/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
S=""
trap 'kill $S 2>/dev/null; restore' EXIT
vram(){ for c in /sys/class/drm/card[0-9]/device; do [ -f $c/mem_info_vram_used ] && printf '%d ' $(( $(cat $c/mem_info_vram_used) / 1048576 )); done; }
state(){ echo "vram used MiB: $(vram); smi count cpu0: $(rdmsr -p 0 -d 0x34 2>/dev/null || echo n/a); CE today: $(journalctl -k --since today --no-pager 2>/dev/null | grep -c 'CE memory read error'); cpu psi: $(head -1 /proc/pressure/cpu | cut -d' ' -f2-4)"; }
modprobe msr 2>/dev/null
log "build $B; idle state: $(state)"
setmax
LD_LIBRARY_PATH=$B/bin:$B/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib $B/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -c 32768 -np 4 --port 8098 \
  --spec-type draft-mtp --spec-draft-n-max 2 -v > $O/server.out 2> $O/server.log < /dev/null &
S=$!
if ! wait_server 8098 $S; then log "server did not come up: $(tail -2 $O/server.log | tr '\n' ' ' | cut -c1-200)"; exit 1; fi
log "loaded: $(state)"
curl -s --max-time 900 http://127.0.0.1:8098/completion -H 'Content-Type: application/json' -d @$W/np4-ab/v100-host-1/req-long.json > $O/long.json
for r in $(seq 1 $NREQ); do
  echo "$(date +%s.%N) request $r start" >> $O/marks.txt
  curl -s --max-time 600 http://127.0.0.1:8098/completion -H 'Content-Type: application/json' -d @$W/np4-ab/v100-host-1/req-short.json > $O/req-$r.json
  echo "$(date +%s.%N) request $r end" >> $O/marks.txt
  log "request $r: $(python3 -I -c 'import json,sys; t=json.load(open(sys.argv[1]))["timings"]; print("prompt %.0f ms, gen %.0f ms = %.2f tok/s, accepted %s/%s" % (t["prompt_ms"], t["predicted_ms"], t["predicted_per_second"], t.get("draft_n_accepted"), t.get("draft_n")))' $O/req-$r.json 2>&1 | tail -1)"
done
log "after: $(state)"
kill $S; wait $S 2>/dev/null; S=""
# per request (launch_slot_ .. release): the three largest gaps between consecutive timestamped log lines
python3 -I - $O/server.log <<'EOF' | tee -a $P
import re, sys
ts = re.compile(r'^(\d+)\.(\d+)\.(\d+)\.(\d+) ')
rows = []
for ln in open(sys.argv[1], errors='replace'):
    m = ts.match(ln)
    if m:
        a, b, c, d = map(int, m.groups())
        rows.append((a * 60 + b + c / 1e3 + d / 1e6, ln.rstrip()[:150]))
starts = [i for i, (t, l) in enumerate(rows) if 'launch_slot_' in l and 'processing task' in l]
ends = [i for i, (t, l) in enumerate(rows) if 'slot      release' in l or 'slot release' in l]
for k, s in enumerate(starts):
    e = next((x for x in ends if x > s), len(rows) - 1)
    gaps = sorted(((rows[i + 1][0] - rows[i][0], i) for i in range(s, e)), reverse=True)[:3]
    print("[stall] task %d: %.2f s, %d lines; largest gaps:" % (k, rows[e][0] - rows[s][0], e - s))
    for g, i in gaps:
        print("[stall]    %.0f ms after: %s" % (g * 1e3, rows[i][1][:120]))
        print("[stall]             before: %s" % rows[i + 1][1][:120])
EOF
log "=== STALL-DIAG DONE ==="
