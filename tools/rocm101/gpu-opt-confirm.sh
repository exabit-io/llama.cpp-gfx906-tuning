#!/bin/bash
# gpu-opt-confirm.sh [NAME ...] -- confirmation pass of the GPU compile-option survey: the variants that pass 1
# (gpu-opt-screen.sh) sorted "up", and the candidates for compiling the package with ROCm 10.1's compiler (XARMS), each
# against the reference in the same blocks. Arms are built as in pass 1: the reference build's programs and host
# libraries (/root/build-master-v100r) on the ROCm 10.1 runtime with ONLY libggml-hip.so taken from the arm (preloaded;
# the reference preloads its own).
#   NAME ...            survey variants, /root/build-gpuopt-NAME
#   XARMS="n=/path/libggml-hip.so ..."   further libraries (builds by the ROCm 10.1 compiler)
# Cells: 4 x 65536 (-ntg 1024) and 1 x 131072 (-ntg 256), N blocks each (default 3), a fresh random arm order per block.
#
# READING, WRITTEN BEFORE THE FIRST CELL. Per cell and metric (decode tok/s per slot, prefill tok/s), against the
# reference's N samples: an arm is "up" when every one of its samples is above every reference sample, "down" when every
# one is below, otherwise "level"; its effect is mean / reference mean - 1.
# (1) A survey variant earns a place in the package only if it is "up" with an effect of at least +0.3 % on one metric in
#     one cell and "down" on no metric in either cell. Its outputs must then reproduce the reference's (perplexity,
#     test-backend-ops) before it ships. Anything else is reported as measured and dropped.
# (2) A ROCm 10.1 compiler arm may replace the ROCm 10.0 compiler for the package only if its effect is not below
#     -0.25 % on any metric in either cell (identical code measured up to 0.16 % apart between n = 3 means in rt-swap);
#     the package built that way must then still pass the design-point acceptance (rocm-ab2.sh). Otherwise the package
#     stays on the ROCm 10.0 compiler.
# (3) If the reference's own samples in a cell span more than 0.5 % on a metric, that metric is not read in that cell.
set -u
W=/root/rocm101-migration-20261114; REF=/root/build-master-v100r; RT=/opt/rocm/core-10.1; N=${N:-3}
P=$W/gpu-opt-confirm.progress
log(){ echo "$(date -Is) [gpuconf] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
ARMS=("ref=$REF:$RT:$REF/bin/libggml-hip.so")
for n in "$@"; do
  lib=/root/build-gpuopt-$n/bin/libggml-hip.so
  [ -f "$lib" ] || { log "$n: no library, skipped"; continue; }
  ARMS+=("$n=$REF:$RT:$lib")
done
for a in ${XARMS:-}; do
  [ -f "${a#*=}" ] || { log "${a%%=*}: no library ${a#*=}, skipped"; continue; }
  ARMS+=("${a%%=*}=$REF:$RT:${a#*=}")
done
k=${#ARMS[@]}; [ "$k" -ge 2 ] || { log "FATAL: nothing to compare"; exit 1; }
log "$k arms, $N blocks: about $(( k * N * 63 / 10 + k * N * 4 )) min"
cd $W || exit 1
SHUFFLE=1 bash $W/screen.sh gpuconf64 65536 4 1024 "$N" "${ARMS[@]}"
SHUFFLE=1 bash $W/screen.sh gpuconf128 131072 1 256 "$N" "${ARMS[@]}"
python3 -I - $W/screen-gpuconf64.tsv $W/screen-gpuconf128.tsv <<'EOF' 2>&1 | tee -a $P
import collections, sys
for path in sys.argv[1:]:
    d = collections.defaultdict(list); cell = '?'
    try: rows = [l.rstrip('\n').split('\t') for l in open(path)]
    except OSError as e: print('no table:', e); continue
    for r in rows: cell = r[0]; d[r[1]].append((float(r[3]), float(r[4])))
    if 'ref' not in d: print(cell, ': no reference samples'); continue
    print('== %s' % cell)
    for i, (metric, fmt) in enumerate((('decode', '%.4f'), ('prefill', '%.1f'))):
        ref = [s[i] for s in d['ref']]; rm = sum(ref) / len(ref); span = (max(ref) - min(ref)) / rm
        print('  %-7s reference n=%d %s (%s-%s)%s' % (metric, len(ref), fmt % rm, fmt % min(ref), fmt % max(ref), '  SPAN %.2f %% > 0.5 %%: not read' % (100 * span) if span > 0.005 else ''))
        for a in sorted(d):
            if a == 'ref': continue
            v = [s[i] for s in d[a]]; m = sum(v) / len(v)
            c = 'up' if min(v) > max(ref) else 'down' if max(v) < min(ref) else 'level'
            print('    %-9s n=%d %s (%s-%s)  %+.2f %%  %s' % (a, len(v), fmt % m, fmt % min(v), fmt % max(v), 100 * (m / rm - 1), c))
EOF
log "=== GPU-OPT CONFIRMATION DONE ==="
