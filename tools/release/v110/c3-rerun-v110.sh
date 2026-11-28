#!/bin/bash
# c3-rerun-v110.sh -- rule (C3) of accept-v110.sh read again on undisturbed samples, with an eviction log.
#
# WHY. accept-v110.sh (2026-11-27 19:03 - 2026-11-28 01:00 UTC, 84 runs, 0 failed) read (C3) from its cell means as
# s100 behind a6 by 0.64 s at 8 x 4096, 0.37 s at 12 x 2048 and 0.27 s at 16 x 2048 per run: over the 0.3 s limit in
# two cells, so by the rule the package stops until that is understood. What the logs of that session show:
#   - the prefill samples of one arm in one cell span 1 to 7 % there (8 x 4096: s100 7.3 %, a6 4.2 %, old 2.3 %),
#     against at most 0.9 % per arm (one arm 0.88 %, every other under 0.5 %; median 0.1 %) in the 22 timing
#     sessions of this cycle before it that have three or more runs per arm (header corrected after the run:
#     it said "at most 0.5 % ... in all 26");
#   - in the slow runs the power log shows the whole pipeline paused: all four dies at 0 % busy and idle power for 5
#     to 20 s, the host idle, 20 to 110 s after the start of the run. It happens in runs of every arm (the v1.0.0
#     build `old` and `a6` as well as s100 and pkg), on both runtimes, in time windows (19:03-19:45, 20:25-21:06,
#     21:36-22:12 UTC) and not outside them (48 x 512 and 8 x 65536: spans 0.03 to 0.17 %).
#   So the cell means of that session cannot resolve 0.3 s of 29 s (1 %), and what disturbed them is not in the
#   builds compared. What pauses the pipeline is not known; KFD's evicted_ms was 2.5 to 3.5 s in the eight-minute
#   test-backend-ops processes of the same evening, so queue eviction is the first candidate.
#
# WHAT. The three cells of (C3), arms s100 (/root/build-final-v110, c782c317d, built-in gate) and a6
# (/root/build-master-v110, 9a7177d0a), ROCm 10.0 runtime, N=5 blocks, through accept-v110.sh with TAG=c3 (same
# screen.sh, same watchdog). Beside it evsample.sh logs every KFD process's evicted_ms once a second.
#
# READING, WRITTEN BEFORE THE FIRST RUN.
#   (V)   A cell is READABLE if the prefill span (max - min over the mean) of each of its two arms is at most 0.5 %.
#   (C3') In a readable cell: s100 behind a6 by 0 to 0.3 s per run (from the cell means) and no MAJOR PENALTY row at
#         the 2 % line => (C3) holds in that cell, and the acceptance's figure for it is put down to the pauses.
#         More than 0.3 s in a readable cell => (C3) fails on clean samples: not published, to the lead.
#         s100 AHEAD of a6 by more than 0.1 s in a readable cell is reported (gate8 and gate9 never read that).
#   (U)   A cell that is not readable is not read. The three cells are then run once more as tag c3b; a cell still
#         not readable there goes to the lead with both sets before anything is published.
#   (E)   Evictions, for the record of the pauses and no rule for the package: per run the growth of evicted_ms and
#         the prefill rate, printed side by side. If the runs more than 0.5 % under their arm's best show at least
#         1000 ms more eviction than that arm's best run, the pauses are KFD queue evictions (what triggers them is
#         then the next question). If slow runs show no such growth, they are something else. No slow run: nothing
#         learned about the pauses here.
# Nothing here is compared with another session by number.
set -u
W=/root/rebase-v110; TAG=${TAG:-c3}; WANT=c782c317dcd40170d3abc51d58c6263663a08fdc
P=$W/c3-rerun-v110.progress; EV=$W/evsample-$TAG.log
log(){ echo "$(date -Is) [c3] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is in progress"; exit 1; }
ls $W/screen-$TAG-*.tsv >/dev/null 2>&1 && { log "FATAL: result files of tag $TAG exist"; exit 1; }
[ "$(systemctl is-active macpro-fand)" = active ] || { log "FATAL: macpro-fand is not active"; exit 1; }
log "start: tag $TAG, load $(cut -d' ' -f1-3 /proc/loadavg), free $(awk '/MemFree/{printf "%.0f GB", $2/1048576}' /proc/meminfo), cached $(awk '/^Cached/{printf "%.0f GB", $2/1048576}' /proc/meminfo), up $(uptime -p)"
bash $W/evsample.sh $EV $EV.stop & SP=$!
TAG=$TAG N=5 WANT=$WANT CELLS="8:4096:512:s100,a6 12:2048:512:s100,a6 16:2048:512:s100,a6" bash $W/accept-v110.sh > $W/c3-rerun-$TAG.out 2>&1
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
    print('[c3] cell %s: per run arm block | prefill tok/s | decode | evicted ms in the process | seconds with >= 100 ms of eviction (offset in the process : ms)' % cell)
    for r in rows: print('[c3]   %-5s b%d | %9.2f | %8.4f | %6d | %s' % r)
    readable = True
    for arm, v in sorted(by.items()):
        pfs = [x[0] for x in v]; mean = sum(pfs) / len(pfs); span = (max(pfs) - min(pfs)) / mean * 100
        best = max(v); slow = [x for x in v if x[0] < best[0] * 0.995]
        if span > 0.5: readable = False
        print('[c3]   %s: n=%d prefill mean %.2f, span %.2f %%; runs more than 0.5 %% under the best: %d; eviction ms of the best run %d, of the slow runs %s'
              % (arm, len(v), mean, span, len(slow), best[1], [x[1] for x in slow]))
    print('[c3] cell %s by (V): %s' % (cell, 'READABLE' if readable else 'NOT READABLE'))
PY
log "=== C3-RERUN $TAG DONE rc=$rc ==="
