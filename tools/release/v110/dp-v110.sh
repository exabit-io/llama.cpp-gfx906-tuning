#!/bin/bash
# dp-v110.sh -- the two design-point cells for llama.cpp v1.1.0 (master-v110): the tile rule's threshold against the
# clang 23 object, and v1.1.0 against v1.0.0. Cells: 4 x 65536 and 1 x 260864, llama-batched-bench, -ntg 1024, N blocks
# (default 3), a fresh random arm order per block (screen.sh, SHUFFLE=1). Kernels run: <16,2> prefill and <1,2> decode.
# Arms (ARMS in the environment, default below; every arm on the ROCm 10.1 runtime):
#   old    /root/build-v101-t85: v1.0.0 master 224b73e24 (the -r2 package's code), compiled by ROCm 10.1, its own
#          programs and libraries
#   c23o   master-v110 compiled by ROCm 10.1 (/root/build-master-v110-101: programs, host libraries, every GPU object)
#          with ONLY the head-256 tile object from ROCm 10.0's clang 23 (tile/lib-c23)
#   tnone, t85, t88, t75   the same with the tile object by clang 24 without a threshold / at that threshold (tile/lib-*)
#
# READING, WRITTEN BEFORE THE FIRST CELL. Per cell and metric (decode tok/s per slot, prefill tok/s): an arm is "up"
# ("down") against another when every one of its samples is above (below) every sample of the other, otherwise
# "level"; its effect is the ratio of the means - 1.
# (1) A tile arm passes at the design points if its effect against c23o is not below -0.25 % on any metric in either
#     cell (the bound of tile-thr-confirm.sh, 2026-11-16).
# (2) What mtp-depth-v110.sh's rule took is confirmed if it passes (1). If it fails and another clean arm passes, that
#     one is taken; if no threshold passes, the rule is dropped (tnone), provided tnone passes (1).
# (3) v1.1.0 against v1.0.0: the arm taken against old, per metric. Within +-2 % is neutral (the acceptance floor of
#     every package so far). A metric where the arm taken is "down" against old by more than 2 % is an upstream cost
#     of the release: it is reported with its size, and the package is not published before the lead has seen it.
# (4) A metric is not read in a cell where c23o's own samples span more than 0.5 %.
set -u
W=/root/rebase-v110; RT=/opt/rocm/core-10.1; NB=/root/build-master-v110-101; N=${N:-3}
ARMS=${ARMS:-"old c23o tnone t88 t75"}; TAGM=${TAGM:-dpv110m}; TAGS=${TAGS:-dpv110s}
P=$W/dp-v110.progress
log(){ echo "$(date -Is) [dpv110] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is in progress"; exit 1; }
for t in $TAGM $TAGS; do [ -s $W/screen-$t.tsv ] && { log "FATAL: screen-$t.tsv exists"; exit 1; }; done
SPECS=()
for a in $ARMS; do
  case $a in
    old) SPECS+=("old=/root/build-v101-t85:$RT");;
    c23o) SPECS+=("c23o=$NB:$RT:$(readlink -f $W/tile/lib-c23/libggml-hip.so)");;
    t*) l=$(readlink -f $W/tile/lib-$a/libggml-hip.so); [ -f "$l" ] || { log "FATAL: no library for $a"; exit 1; }; SPECS+=("$a=$NB:$RT:$l");;
    *) log "FATAL: unknown arm $a"; exit 1;;
  esac
done
k=${#SPECS[@]}
log "arms: $ARMS; $N blocks: about $(( k * N * 63 / 10 + k * N * 113 / 10 )) min; master-v110 $(git -C /root/wt-master-v110 rev-parse --short HEAD)"
cd $W || exit 1
SHUFFLE=1 bash $W/screen.sh $TAGM 65536 4 1024 "$N" "${SPECS[@]}"
SHUFFLE=1 bash $W/screen.sh $TAGS 260864 1 1024 "$N" "${SPECS[@]}"
python3 -I - $W/screen-$TAGM.tsv $W/screen-$TAGS.tsv <<'PY' 2>&1 | tee -a $P
import collections, sys
for path in sys.argv[1:]:
    d = collections.defaultdict(list); cell = '?'
    try: rows = [l.rstrip('\n').split('\t') for l in open(path)]
    except OSError as e: print('no table:', e); continue
    for r in rows:
        try: d[r[1]].append((float(r[3]), float(r[4]))); cell = r[0]
        except (ValueError, IndexError): print('  bad row:', r)
    print('== %s' % cell)
    for i, (metric, fmt) in enumerate((('decode', '%.4f'), ('prefill', '%.1f'))):
        def st(a): v = [s[i] for s in d[a]]; return v, sum(v) / len(v)
        def cls(v, w): return 'up' if min(v) > max(w) else 'down' if max(v) < min(w) else 'level'
        refs = [a for a in ('c23o', 'old') if a in d]
        note = ''
        if 'c23o' in d:
            v, m = st('c23o'); span = (max(v) - min(v)) / m
            if span > 0.005: note = '  C23O SPAN %.2f %% > 0.5 %%: NOT READ' % (100 * span)
        print('  %s%s' % (metric, note))
        for a in d:
            v, m = st(a)
            s = ['%-6s n=%d %s (%s-%s)' % (a, len(v), fmt % m, fmt % min(v), fmt % max(v))]
            for r in refs:
                if r == a: continue
                w, mw = st(r); s.append('against %s %+.2f %% %s' % (r, 100 * (m / mw - 1), cls(v, w)))
            print('    ' + '  '.join(s))
PY
log "=== DP-V110 DONE ==="
