#!/bin/bash
# fp8-kv.sh -- an FP8 (f8_e4m3) KV cache at the multi-user design point, and whether ROCm 10.1's compiler costs it.
#
# Why: the report's open list has "not measured on ROCm 10.1: FP8 KV caches". The FP8 cache came with llama.cpp v1.0.0
# and has never been timed on this box on a real model. codegen/loopscan.py (2026-11-16) shows that clang 24 puts far
# more scratch reloads inside the loops of the flash-attention vec kernels with an FP8 V cache than clang 23 does
# (flash_attn_ext_vec<256,2,f8_e4m3,f8_e4m3>: 146 -> 623 in loops), and that this is not the machine-LICM bug of the
# tile kernel (the count stays with -disable-machine-licm and with every VGPR threshold tried).
#
# Arms: the r1 build's programs and host libraries (/root/build-v101-t90, master 290fc642a, gcc host code) on the ROCm
# 10.1 runtime, with ONLY libggml-hip.so chosen per arm (preloaded by full path; r1 preloads its own):
#   c23f8   GPU library compiled by ROCm 10.0's clang 23 (/root/build-master-v100r, df2c6e5c5: the same sources, the two
#           commits differ only in the CMake rule that is inert for clang 23), -ctk f8_e4m3 -ctv f8_e4m3
#   r1f8    GPU library compiled by ROCm 10.1's clang 24 with the tile rule = the package's GPU code, FP8 / FP8
#   r1f16   the same library with the f16 / f16 cache of every other cell of the report (what FP8 costs or gains)
# Cell: 4 x 65536, -ntg 1024, N blocks (default 3), a fresh random arm order per block (screen.sh).
#
# READING, WRITTEN BEFORE THE FIRST CELL. Metrics: decode tok/s per slot, prefill tok/s. An arm is "down" ("up") against
# another when every one of its samples is below (above) every sample of the other; otherwise "level".
# (1) Does clang 24 cost the FP8 cache? Yes on a metric if r1f8 is "down" against c23f8 there; the size is the
#     difference of the means. "level" or "up": no cost, the static count has no effect that this cell shows.
# (2) What the FP8 cache is against f16 on the package's code: r1f8 against r1f16, both metrics, reported as measured.
# (3) A per-object compile option for the FP8 vec kernels is looked for only if (1) is yes with 1 % or more on a metric
#     AND the FP8 cache is a configuration worth serving: decode of the better FP8 arm not more than 10 % under r1f16.
#     Otherwise the finding is recorded with its size and nothing is changed (no profile in settings/ uses an FP8 cache).
# (4) A cell whose log shows an attention fallback, or that fails, is discarded and reported; an arm with fewer than
#     N samples is not read.
set -u
W=/root/rocm101-migration-20261114; B=/root/build-v101-t90; RT=/opt/rocm/core-10.1; N=${N:-3}; TAG=${TAG:-fp8kv}
P=$W/fp8-kv.progress
log(){ echo "$(date -Is) [fp8] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
[ -s $W/screen-$TAG.tsv ] && { log "FATAL: screen-$TAG.tsv exists"; exit 1; }
C23=$(readlink -f /root/build-master-v100r/bin/libggml-hip.so); R1=$(readlink -f $B/bin/libggml-hip.so)
[ -f "$C23" ] && [ -f "$R1" ] || { log "FATAL: a GPU library is missing"; exit 1; }
for l in "$C23" "$R1"; do
  log "$l: producer $(readelf -p .comment "$l" 2>/dev/null | grep -oE 'clang version [0-9.]+' | sort -u | tr '\n' ';'), $(stat -c %s "$l") bytes"
done
cd $W || exit 1
log "3 arms, $N blocks, 4 x 65536"
SHUFFLE=1 KVK=f8_e4m3 KVV=f8_e4m3 ARMKV="r1f16=f16/f16" bash $W/screen.sh $TAG 65536 4 1024 "$N" \
  "c23f8=$B:$RT:$C23" "r1f8=$B:$RT:$R1" "r1f16=$B:$RT:$R1"
python3 -I - $W/screen-$TAG.tsv "$N" <<'EOF' 2>&1 | tee -a $P
import collections, sys
d = collections.defaultdict(list)
try: rows = [l.rstrip('\n').split('\t') for l in open(sys.argv[1])]
except OSError as e: print('[fp8] no table:', e); sys.exit()
for r in rows: d[r[1]].append((float(r[3]), float(r[4])))
N = int(sys.argv[2])
def rel(a, b): return 'up' if min(a) > max(b) else 'down' if max(a) < min(b) else 'level'
for i, (metric, fmt) in enumerate((('decode', '%.4f'), ('prefill', '%.1f'))):
    v = {a: [s[i] for s in d[a]] for a in d}
    for a in ('c23f8', 'r1f8', 'r1f16'):
        if a in v: print('[fp8] %-7s %-6s n=%d %s (%s-%s)%s' % (metric, a, len(v[a]), fmt % (sum(v[a]) / len(v[a])), fmt % min(v[a]), fmt % max(v[a]), '' if len(v[a]) >= N else '  FEWER THAN N: not read'))
    for a, b in (('r1f8', 'c23f8'), ('r1f8', 'r1f16'), ('c23f8', 'r1f16')):
        if len(v.get(a, [])) >= N and len(v.get(b, [])) >= N:
            print('[fp8] %-7s %s against %s: %s, %+.2f %%' % (metric, a, b, rel(v[a], v[b]), 100 * (sum(v[a]) / len(v[a]) / (sum(v[b]) / len(v[b])) - 1)))
EOF
log "=== FP8 KV DONE ==="
