#!/bin/bash
# gpu-opt24-confirm.sh -- confirmation pass of the clang-24 GPU compile-option survey (gpu-opt24-screen.sh, 2026-11-16/17).
# That survey's rule (3) sends two variants here: o2 ("up" on prefill in both passes: +0.20 % at 4 x 64K, +0.57 % at
# 1 x 255K, one sample each) and v80 ("up" on prefill at 4 x 64K, +0.33 %; level at 1 x 255K). Neither was "down" anywhere.
#
# The survey was built on master 290fc642a (the -r1 package: tile rule at threshold 90). While it ran, the package moved
# to 224b73e24 (-r2: threshold 85, report s.5.11), and 85 itself read +0.34 % / +0.40 % prefill over 90 at the
# design-point kernels (tile-thr-confirm.sh). So the bar for a place in the package is no longer the survey's reference
# but the shipped rule, and "-O2" has to be read on the shipped tree as well. Arms, all the reference build's programs
# and host libraries (/root/build-v101-t90) on the ROCm 10.1 runtime with ONLY libggml-hip.so chosen per arm (preloaded
# by full path), every library compiled by ROCm 10.1's clang 24:
#   t90    /root/build-v101-t90          290fc642a, the survey's reference (threshold 90 on the head-256 tile object)
#   t85    /root/build-v101-t85          224b73e24, the shipped rule (threshold 85 on that object), built like t90
#   o2     /root/build-gpuopt24-o2       290fc642a with the HIP sources at -O2 instead of -O3 (the survey's variant)
#   o2t85  /root/build-gpuopt24-o2t85    224b73e24 with the HIP sources at -O2 (the form in which -O2 could ship)
#   v80    /root/build-gpuopt24-v80      df2c6e5c5 with threshold 80 on EVERY HIP source (the survey's variant)
# Cells: the survey's, 4 x 65536 and 1 x 260864, -ntg 1024, N blocks each (default 3), a fresh random arm order per block.
#
# READING, WRITTEN BEFORE THE FIRST CELL. Per cell and metric (decode tok/s per slot, prefill tok/s): an arm is "up"
# ("down") against another when every one of its samples is above (below) every sample of the other, otherwise "level";
# its effect is the ratio of the means - 1.
# (1) The survey's sorting is checked against t90: o2 and v80 are reported "up"/"level"/"down" against it. That alone
#     earns nothing.
# (2) A place in the package is earned only against t85: o2t85 or v80 qualifies if it is "up" against t85 with an effect
#     of at least +0.3 % on one metric in one cell and "down" against t85 on no metric in either cell (the bound of
#     gpu-opt-confirm.sh's rule (1)). o2 is not a shipping form (its tree has threshold 90, which costs MTP decode at
#     depth); it is here to separate what -O2 does from what the threshold does.
# (3) A qualifying arm is still only a candidate: it changes every kernel of every HIP object, so before a package it
#     needs loopscan.py/tilescan.py on every device function, the MTP cell at depth (mtp-depth2.sh), its outputs
#     reproduced (perplexity, test-backend-ops) and both package acceptances. If nothing qualifies, the package's GPU
#     flags stay as they are and the survey is closed for this compiler.
# (4) If t85's own samples in a cell span more than 0.5 % on a metric, that metric is not read in that cell.
set -u
W=/root/rocm101-migration-20261114; REF=/root/build-v101-t90; RT=/opt/rocm/core-10.1; N=${N:-3}
P=$W/gpu-opt24-confirm.progress
log(){ echo "$(date -Is) [gpuconf24] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x llama-perplexit >/dev/null && { log "FATAL: a perplexity run is in progress"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is in progress"; exit 1; }
for t in gpuconf24m gpuconf24s; do [ -s $W/screen-$t.tsv ] && { log "FATAL: screen-$t.tsv exists"; exit 1; }; done
declare -A B=([t90]=/root/build-v101-t90 [t85]=/root/build-v101-t85 [o2]=/root/build-gpuopt24-o2 [o2t85]=/root/build-gpuopt24-o2t85 [v80]=/root/build-gpuopt24-v80)
ARMS=()
for a in t90 t85 o2 o2t85 v80; do
  lib=$(readlink -f ${B[$a]}/bin/libggml-hip.so)
  [ -f "$lib" ] || { log "FATAL: $a has no library in ${B[$a]}"; exit 1; }
  ARMS+=("$a=$REF:$RT:$lib")
done
# what the arms are, read back from the builds (objects whose GPU code differs from t90's and from t85's)
for a in t85 o2 o2t85 v80; do
  log "$a against t90: $(python3 -I $W/codegen/devcmp.py $REF ${B[$a]} | grep -oE 'different in [0-9]+[^,;]*' | head -1); against t85: $(python3 -I $W/codegen/devcmp.py ${B[t85]} ${B[$a]} | grep -oE 'different in [0-9]+[^,;]*' | head -1)"
done
k=${#ARMS[@]}
log "$k arms, $N blocks: about $(( k * N * 63 / 10 + k * N * 113 / 10 )) min"
cd $W || exit 1
SHUFFLE=1 bash $W/screen.sh gpuconf24m 65536 4 1024 "$N" "${ARMS[@]}"
SHUFFLE=1 bash $W/screen.sh gpuconf24s 260864 1 1024 "$N" "${ARMS[@]}"
python3 -I - $W/screen-gpuconf24m.tsv $W/screen-gpuconf24s.tsv <<'EOF' 2>&1 | tee -a $P
import collections, sys
qual = collections.defaultdict(list); down = collections.defaultdict(list)
for path in sys.argv[1:]:
    d = collections.defaultdict(list); cell = '?'
    try: rows = [l.rstrip('\n').split('\t') for l in open(path)]
    except OSError as e: print('no table:', e); continue
    for r in rows:
        try: d[r[1]].append((float(r[3]), float(r[4]))); cell = r[0]
        except (ValueError, IndexError): print('  bad row:', r)
    if 't85' not in d or 't90' not in d: print(cell, ': no t85 or t90 samples: NOT READ'); continue
    print('== %s' % cell)
    for i, (metric, fmt) in enumerate((('decode', '%.4f'), ('prefill', '%.1f'))):
        def st(a): v = [s[i] for s in d[a]]; return v, sum(v) / len(v)
        def cls(v, w): return 'up' if min(v) > max(w) else 'down' if max(v) < min(w) else 'level'
        b90, m90 = st('t90'); b85, m85 = st('t85'); span = (max(b85) - min(b85)) / m85; read = span <= 0.005
        print('  %-7s t90 n=%d %s (%s-%s)   t85 n=%d %s (%s-%s), %+.2f %% %s against t90%s' % (
            metric, len(b90), fmt % m90, fmt % min(b90), fmt % max(b90), len(b85), fmt % m85, fmt % min(b85), fmt % max(b85),
            100 * (m85 / m90 - 1), cls(b85, b90), '' if read else '  T85 SPAN %.2f %% > 0.5 %%: not read' % (100 * span)))
        for a in ('o2', 'o2t85', 'v80'):
            if a not in d: print('    %-6s no samples' % a); continue
            v, m = st(a); c90, c85 = cls(v, b90), cls(v, b85); e85 = m / m85 - 1
            print('    %-6s n=%d %s (%s-%s)  against t90 %+.2f %% %-5s  against t85 %+.2f %% %s' % (
                a, len(v), fmt % m, fmt % min(v), fmt % max(v), 100 * (m / m90 - 1), c90, 100 * e85, c85))
            if read and c85 == 'up' and e85 >= 0.003: qual[a].append('%s %s %+.2f %%' % (cell, metric, 100 * e85))
            if read and c85 == 'down': down[a].append('%s %s %+.2f %%' % (cell, metric, 100 * e85))
print('== rule (2), against t85')
for a in ('o2t85', 'v80'):
    if qual[a] and not down[a]: print('  %-6s QUALIFIES as a candidate: up in %s' % (a, '; '.join(qual[a])))
    else: print('  %-6s does not qualify: up by 0.3 %% or more in %s; down in %s' % (a, '; '.join(qual[a]) or 'nothing', '; '.join(down[a]) or 'nothing'))
print('  o2     (not a shipping form) up by 0.3 %% or more against t85 in %s; down in %s' % ('; '.join(qual['o2']) or 'nothing', '; '.join(down['o2']) or 'nothing'))
EOF
log "=== GPU-OPT24 CONFIRMATION DONE ==="
