#!/bin/bash
# dp-rerun-v110.sh -- the two design points, package against v1.0.0, read again on undisturbed samples.
#
# WHY. accept-v110.sh read (P2) as met: no MAJOR PENALTY row of pkg against old at 4 x 65536 and 1 x 260864. But the
# prefill samples of those two cells are the ones the evening's pauses hit hardest (per-arm spread 1.8 to 7.4 % at
# 4 x 65536, 1.0 to 2.7 % at 1 x 260864), so their prefill rows have intervals of [-5.6, +8.3] and [-2.0, +1.1]: they
# could not have shown a 2 % penalty. c3-rerun-v110.sh introduced the rule that a prefill row is read only where every
# arm spreads at most 0.5 %; by that rule these two prefill rows were not readable either. The package is published
# (2026-11-28 02:58 UTC) on the decode rows of these cells (spread 0.06 to 0.21 %, level), the clean 8 x 65536 cell
# (prefill -0.14 % [-0.18, -0.10]), the tile-object session dp-v110 (-0.5 to +0.3 % at both points) and MTP at depth
# (prefill -0.14 %). This run is the direct reading that was missing.
#
# WHAT. 4 x 65536 and 1 x 260864, -ntg 1024, arms pkg (installed package, from the channel) and old (v1.0.0 master
# 224b73e24 by ROCm 10.1), N=3 blocks, through accept-v110.sh with TAG=dp2; evsample.sh beside it. About 105 min.
#
# READING, WRITTEN BEFORE THE FIRST RUN.
#   (V)   A cell is READABLE if the prefill span (max - min over the mean) of each arm is at most 0.5 %.
#   (P2') In a readable cell, pkg against old (sub-stats.py, 2 % line): no MAJOR PENALTY row => (P2) holds on clean
#         samples for that cell; the effect and interval replace the acceptance's in the report (the published notes
#         get the tighter numbers as an edit of the release text, nothing else changes).
#         A MAJOR PENALTY row => a cost of 1.1.0 against 1.0.0 at a design point that the acceptance could not see:
#         to the lead at once, with the numbers (the package is out; 1.0.0 is still in the channel).
#         A significant difference that is not MAJOR is reported with its size.
#   (U)   A cell that is not readable is not read; it is run once more (tag dp3); still not readable => reported as
#         such with the eviction log, and the published wording (wide interval, cause named) stands.
#   (E)   Evictions as in c3-rerun-v110.sh: per run the growth of evicted_ms beside the prefill rate; slow runs
#         (more than 0.5 % under their arm's best) with at least 1000 ms more eviction than the best run => the pauses
#         are queue evictions.
# Nothing here is compared with another session by number.
set -u
W=/root/rebase-v110; TAG=${TAG:-dp2}; WANT=c782c317dcd40170d3abc51d58c6263663a08fdc
P=$W/dp-rerun-v110.progress; EV=$W/evsample-$TAG.log
log(){ echo "$(date -Is) [dp] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is in progress"; exit 1; }
ls $W/screen-$TAG-*.tsv >/dev/null 2>&1 && { log "FATAL: result files of tag $TAG exist"; exit 1; }
[ "$(systemctl is-active macpro-fand)" = active ] || { log "FATAL: macpro-fand is not active"; exit 1; }
log "start: tag $TAG, load $(cut -d' ' -f1-3 /proc/loadavg), free $(awk '/MemFree/{printf "%.0f GB", $2/1048576}' /proc/meminfo), cached $(awk '/^Cached/{printf "%.0f GB", $2/1048576}' /proc/meminfo), up $(uptime -p)"
bash $W/evsample.sh $EV $EV.stop & SP=$!
TAG=$TAG N=3 WANT=$WANT CELLS="4:65536:1024:pkg,old 1:260864:1024:pkg,old" bash $W/accept-v110.sh > $W/dp-rerun-$TAG.out 2>&1
rc=$?
touch $EV.stop; wait $SP 2>/dev/null
log "accept-v110.sh TAG=$TAG ended rc=$rc"
python3 -I - $W $TAG $EV <<'PY' 2>&1 | tee -a $P
import collections, glob, re, sys, datetime as dt
w, tag, evf = sys.argv[1:4]
# eviction samples: pid -> [(epoch, ms)]
ev = collections.defaultdict(list)
for l in open(evf):
    if l.startswith('#'): continue
    p = l.split()
    if len(p) == 5 and p[3].startswith('llama-batched'): ev[p[2]].append((int(p[0]), int(p[4])))
procs = sorted(((v[0][0], v[-1][0], v[-1][1] - v[0][1], v[-1][1], pid, v) for pid, v in ev.items()))
for path in sorted(glob.glob('%s/screen-%s-*.progress' % (w, tag))):
    cell = path.split('screen-%s-' % tag)[1][:-9]
    runs = []; prev = None
    for l in open(path):
        m = re.match(r'(\S+) \[screen\] (.*)', l)
        if not m: continue
        try: t = dt.datetime.fromisoformat(m.group(1)).timestamp()
        except ValueError: continue
        body = m.group(2)
        if prev is None and 'order' in body: prev = t
        m2 = re.match(r'(.+?) (\S+) b(\d+): decode ([\d.]+), prefill ([\d.]+)', body)
        if m2 and prev is not None:
            runs.append((prev, t, m2.group(2), int(m2.group(3)), float(m2.group(4)), float(m2.group(5)))); prev = t
    by = collections.defaultdict(list)
    rows = []
    for a, b, arm, blk, dec, pf in runs:
        mine = [x for x in procs if x[0] >= a - 2 and x[1] <= b + 2]
        tot = sum(x[3] for x in mine) if mine else -1
        steps = []
        for x in mine:
            v = x[5]
            for i in range(1, len(v)):
                if v[i][1] - v[i-1][1] >= 100: steps.append('+%ds:%d' % (v[i][0] - v[0][0], v[i][1] - v[i-1][1]))
        by[arm].append((pf, tot)); rows.append((arm, blk, pf, dec, tot, ' '.join(steps)))
    print('[dp] cell %s: per run arm block | prefill tok/s | decode | evicted ms in the process | seconds with >= 100 ms of eviction (offset in the process : ms)' % cell)
    for r in rows: print('[dp]   %-5s b%d | %9.2f | %8.4f | %6d | %s' % r)
    readable = True
    for arm, v in sorted(by.items()):
        pfs = [x[0] for x in v]; mean = sum(pfs) / len(pfs); span = (max(pfs) - min(pfs)) / mean * 100
        best = max(v); slow = [x for x in v if x[0] < best[0] * 0.995]
        if span > 0.5: readable = False
        print('[dp]   %s: n=%d prefill mean %.2f, span %.2f %%; runs more than 0.5 %% under the best: %d; eviction ms of the best run %d, of the slow runs %s'
              % (arm, len(v), mean, span, len(slow), best[1], [x[1] for x in slow]))
    print('[dp] cell %s by (V): %s' % (cell, 'READABLE' if readable else 'NOT READABLE'))
PY
log "=== DP-RERUN $TAG DONE rc=$rc ==="
