#!/bin/bash
# gpu-opt-screen.sh [NAME ...] -- first timing pass of the GPU compile-option survey (gpu-opt-builds.sh): one 4 x 64K
# cell per variant, the reference three times (first, middle, last) to show drift and noise. Every arm is the reference
# build's programs and host libraries (/root/build-master-v100r) on the ROCm 10.1 runtime with ONLY libggml-hip.so taken
# from the variant (preloaded; the reference arms preload their own). Variants whose GPU machine code is byte-identical
# to the reference (codegen/devcmp.py) are listed and not timed. Default: every /root/build-gpuopt-* that built.
#
# READING, WRITTEN BEFORE THE FIRST CELL. One sample per variant, so this pass only sorts: with the three reference
# samples spanning [lo, hi] on a metric, a variant is "up" on it when above hi by more than the span (at least 0.05
# tok/s decode, 1.0 tok/s prefill), "down" when below lo by the same margin, otherwise "level". (1) schdef (the
# compiler's default scheduler; the source tree puts it 2.5 % below the reference on decode) must come out "down" on
# decode, or the pass cannot see an effect of that size and nothing is read. (2) Variants "up" on decode or prefill and
# not "down" on the other go to the confirmation pass (n = 3 blocks, shuffled, with the reference) and only that pass
# can earn a place in the package; everything else is reported as measured and dropped. (3) If the three reference
# samples drift monotonically by more than 0.15 tok/s decode, the pass is repeated in two halves.
set -u
W=/root/rocm101-migration-20261114; REF=/root/build-master-v100r; RT=/opt/rocm/core-10.1; TAG=${TAG:-gpuopt}
P=$W/gpu-opt-screen.progress
log(){ echo "$(date -Is) [gpuopt] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
V=()
for n in ${*:-$(ls -d /root/build-gpuopt-* 2>/dev/null | sed 's|.*/build-gpuopt-||')}; do
  lib=/root/build-gpuopt-$n/bin/libggml-hip.so
  [ -f "$lib" ] || { log "$n: no library, skipped"; continue; }
  d=$(python3 -I $W/codegen/devcmp.py $REF /root/build-gpuopt-$n | grep -oE 'different in [0-9]+' | grep -oE '[0-9]+')
  [ -n "$d" ] || { log "$n: comparison failed, skipped"; continue; }
  if [ "$d" = 0 ]; then log "$n: GPU code identical to the reference in every object, not timed"; continue; fi
  log "$n: GPU code differs in $d objects"
  V+=("$n")
done
k=${#V[@]}; [ "$k" -ge 1 ] || { log "FATAL: nothing to time"; exit 1; }
h=$(( (k + 1) / 2 )); ARMS=("base1=$REF:$RT:$REF/bin/libggml-hip.so")
for i in "${!V[@]}"; do
  [ "$i" = "$h" ] && ARMS+=("base2=$REF:$RT:$REF/bin/libggml-hip.so")
  ARMS+=("${V[$i]}=$REF:$RT:/root/build-gpuopt-${V[$i]}/bin/libggml-hip.so")
done
# XARMS="name=/path/libggml-hip.so ..." (optional): further libraries timed the same way after the variants, for
# information only (not part of the reading above), e.g. builds by the ROCm 10.1 compiler.
x=0
for a in ${XARMS:-}; do
  [ -f "${a#*=}" ] || { log "${a%%=*}: no library ${a#*=}, skipped"; continue; }
  ARMS+=("${a%%=*}=$REF:$RT:${a#*=}"); x=$((x + 1)); log "${a%%=*}: extra arm ${a#*=}"
done
ARMS+=("base3=$REF:$RT:$REF/bin/libggml-hip.so")
log "$k variants + $x extra + 3 reference cells, about $(( (k + x + 3) * 63 / 10 )) min"
cd $W && bash $W/screen.sh $TAG 65536 4 1024 1 "${ARMS[@]}"
python3 -I - $W/screen-$TAG.tsv <<'EOF' 2>&1 | tee -a $P
import sys
rows = [l.rstrip('\n').split('\t') for l in open(sys.argv[1])]
d = {r[1]: (float(r[3]), float(r[4])) for r in rows}
base = [d[k] for k in ('base1', 'base2', 'base3') if k in d]
if len(base) < 3: sys.exit('fewer than three reference cells: not read')
def band(i, floor):
    lo, hi = min(b[i] for b in base), max(b[i] for b in base); m = max(hi - lo, floor); return lo, hi, m
bd, bp = band(0, 0.05), band(1, 1.0)
def cls(v, b): return 'up' if v > b[1] + b[2] else 'down' if v < b[0] - b[2] else 'level'
md = sum(b[0] for b in base) / 3; mp = sum(b[1] for b in base) / 3
print('reference: decode %s (mean %.4f), prefill %s (mean %.1f)' % (' '.join('%.4f' % b[0] for b in base), md, ' '.join('%.1f' % b[1] for b in base), mp))
mono = (base[0][0] < base[1][0] < base[2][0]) or (base[0][0] > base[1][0] > base[2][0])
if mono and abs(base[2][0] - base[0][0]) > 0.15: print('REFERENCE DRIFTS monotonically by %.3f tok/s: repeat in two halves' % (base[2][0] - base[0][0]))
keep = []
for k, (dec, pre) in sorted(d.items(), key=lambda kv: -kv[1][0]):
    if k.startswith('base'): continue
    cd, cp = cls(dec, bd), cls(pre, bp)
    if ('up' in (cd, cp)) and 'down' not in (cd, cp): keep.append(k)
    print('  %-9s decode %.4f (%+.2f %%) %-5s  prefill %.1f (%+.2f %%) %-5s' % (k, dec, 100 * (dec / md - 1), cd, pre, 100 * (pre / mp - 1), cp))
if 'schdef' in d and cls(d['schdef'][0], bd) != 'down': print('POSITIVE CONTROL schdef is not down on decode: the pass is not read')
print('to the confirmation pass:', ' '.join(keep) or 'none')
EOF
log "=== GPU-OPT SCREEN PASS 1 DONE ==="
