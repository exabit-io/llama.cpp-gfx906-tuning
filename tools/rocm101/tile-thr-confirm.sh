#!/bin/bash
# tile-thr-confirm.sh -- two-cell confirmation of a replacement for the head-256 tile rule's threshold.
#
# Why: mtp-depth2.sh (2026-11-16, n = 3, 117.7K-token depth, --spec-draft-n-max 3) read, under its header's rule:
#   (1) the shipped threshold 90 costs MTP decode: 45.481 tok/s against 46.332 with no rule, "down", -1.84 %
#   (2) thresholds 88 and 85 qualify: MTP decode 48.585 and 48.357 ("up" against no rule, +4.86 % and +4.37 %),
#       prefill 611.4 and 612.2 tok/s ("up" against clang 23's 608.5)
#   (3) the replacement is the one with the higher prefill mean unless they tie within 0.1 %: 85 (+0.12 % over 88).
#       88 was 0.47 % above 85 on MTP decode in that session; that was not the rule's criterion and is reported.
# That rule sends the replacement through this pass (gpu-opt-confirm.sh's cells and rule) before a package is built.
#
# Arms: the reference build's programs and host libraries (/root/build-master-v100r) on the ROCm 10.1 runtime with ONLY
# libggml-hip.so chosen per arm, preloaded by full path:
#   ref   the reference build's own library (ROCm 10.0's clang 23, every object)
#   v90   ROCm 10.1 build's objects with the tile object compiled at threshold 90 (/root/swap/v90: the shipped rule)
#   v85   the same with threshold 85 (the candidate)
#   v88   the same with threshold 88 (the fallback)
# Cells: 4 x 65536 (-ntg 1024) and 1 x 131072 (-ntg 256), N blocks each (default 3), a fresh random arm order per block.
#
# READING, WRITTEN BEFORE THE FIRST CELL. Per cell and metric (decode tok/s per slot, prefill tok/s): an arm is "up"
# ("down") against another when every one of its samples is above (below) every sample of the other, otherwise
# "level"; its effect is the ratio of the means - 1.
# (1) A threshold passes if its effect against ref is not below -0.25 % on any metric in either cell (the bound of
#     gpu-opt-confirm.sh's rule (2): identical code measured up to 0.16 % apart between n = 3 means) AND it is not
#     "down" against v90 with an effect below -0.25 % on any metric in either cell (it must not give back what the
#     shipped rule has at the two design-point kernels).
# (2) 85 is taken if it passes. If 85 fails and 88 passes, 88 is taken. If neither passes, the rule stays at 90 and
#     the MTP cost is recorded as open.
# (3) If ref's own samples span more than 0.5 % on a metric in a cell, that metric is not read in that cell.
# (4) The threshold taken goes into the fork (one number in the CMake rule), and the package built from it must pass
#     the design-point acceptance on the installed deb AND an MTP-at-depth cell against the -r1 library before it ships.
set -u
W=/root/rocm101-migration-20261114; REF=/root/build-master-v100r; RT=/opt/rocm/core-10.1; N=${N:-3}
P=$W/tile-thr-confirm.progress
log(){ echo "$(date -Is) [thrconf] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
for t in thr64 thr128; do [ -s $W/screen-$t.tsv ] && { log "FATAL: screen-$t.tsv exists"; exit 1; }; done
ARMS=("ref=$REF:$RT:$(readlink -f $REF/bin/libggml-hip.so)")
for a in v90 v85 v88; do
  l=$(readlink -f /root/swap/$a/libggml-hip.so); [ -f "$l" ] || { log "FATAL: no library for $a"; exit 1; }
  ARMS+=("$a=$REF:$RT:$l")
done
log "${#ARMS[@]} arms, $N blocks: about $(( ${#ARMS[@]} * N * 63 / 10 + ${#ARMS[@]} * N * 4 )) min"
cd $W || exit 1
SHUFFLE=1 bash $W/screen.sh thr64 65536 4 1024 "$N" "${ARMS[@]}"
SHUFFLE=1 bash $W/screen.sh thr128 131072 1 256 "$N" "${ARMS[@]}"
python3 -I - "$N" $W/screen-thr64.tsv $W/screen-thr128.tsv <<'EOF' 2>&1 | tee -a $P
import collections, sys
N = int(sys.argv[1]); ok = {'v85': True, 'v88': True}
def rel(a, b): return 'up' if min(a) > max(b) else 'down' if max(a) < min(b) else 'level'
for path in sys.argv[2:]:
    d = collections.defaultdict(list); cell = '?'
    try: rows = [l.rstrip('\n').split('\t') for l in open(path)]
    except OSError as e: print('[thrconf] no table:', e); ok = {k: False for k in ok}; continue
    for r in rows: cell = r[0]; d[r[1]].append((float(r[3]), float(r[4])))
    print('[thrconf] == %s' % cell)
    for a in ('ref', 'v90', 'v85', 'v88'):
        if len(d[a]) < N: print('[thrconf]   %s has %d of %d samples: cell not read' % (a, len(d[a]), N)); ok = {k: False for k in ok}
    if any(len(d[a]) < N for a in ('ref', 'v90', 'v85', 'v88')): continue
    for i, (metric, fmt) in enumerate((('decode', '%.4f'), ('prefill', '%.1f'))):
        v = {a: [s[i] for s in d[a]] for a in d}; m = {a: sum(v[a]) / len(v[a]) for a in v}
        span = (max(v['ref']) - min(v['ref'])) / m['ref']; read = span <= 0.005
        print('[thrconf]   %-7s ref n=%d %s (%s-%s)%s' % (metric, len(v['ref']), fmt % m['ref'], fmt % min(v['ref']), fmt % max(v['ref']), '' if read else '  SPAN %.2f %% > 0.5 %%: not read' % (100 * span)))
        for a in ('v90', 'v85', 'v88'):
            e = 100 * (m[a] / m['ref'] - 1); line = '[thrconf]     %-4s n=%d %s (%s-%s)  against ref %+.2f %% %s' % (a, len(v[a]), fmt % m[a], fmt % min(v[a]), fmt % max(v[a]), e, rel(v[a], v['ref']))
            if a != 'v90':
                e90 = 100 * (m[a] / m['v90'] - 1); r90 = rel(v[a], v['v90']); line += '; against v90 %+.2f %% %s' % (e90, r90)
                if read and (e < -0.25 or (r90 == 'down' and e90 < -0.25)): ok[a] = False; line += '  FAILS'
            print(line)
print('[thrconf] verdict: v85 %s, v88 %s -> %s' % ('passes' if ok['v85'] else 'fails', 'passes' if ok['v88'] else 'fails',
      'threshold 85' if ok['v85'] else 'threshold 88' if ok['v88'] else 'the rule stays at 90'))
EOF
log "=== THRESHOLD CONFIRMATION DONE ==="
