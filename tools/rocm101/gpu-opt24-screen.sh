#!/bin/bash
# gpu-opt24-screen.sh [NAME ...] -- timing passes of the second GPU compile-option survey (gpu-opt24-builds.sh: the
# variants compiled by ROCm 10.1's clang 24 on master 290fc642a), at BOTH design points: 4 x 64K (pass M) and 1 x 255K
# (pass S), -ntg 1024, one cell per variant, the reference three times per pass (first, middle, last). Every arm is the
# reference build's programs and host libraries (/root/build-v101-t90, the -r1 package's code) on the ROCm 10.1 runtime
# with ONLY libggml-hip.so taken from the variant (preloaded; the reference arms preload their own). Variants whose GPU
# machine code is byte-identical to the reference (codegen/devcmp.py) are listed and not timed.
# Default: every /root/build-gpuopt24-* that built, plus v90 (/root/build-v101-v90: the threshold on every source).
# Two further arms are positive controls with a sign known for this compiler:
#   trk       -amdgpu-use-amdgpu-trackers=1: decode down (3.1 % at 4 x 64K under clang 23, 5.0 % at 1 x 128K under both)
#   c24plain  /root/build-master-v101, clang 24 without the tile rule: prefill down (1.3 % at 4 x 64K, 4.1 % at 1 x 255K)
#
# READING, WRITTEN BEFORE THE FIRST CELL. One sample per variant and cell, so the passes only sort. In each pass, with
# the three reference samples spanning [lo, hi] on a metric, a variant is "up" on it when above hi by more than the
# span (at least 0.05 tok/s decode, 1.0 tok/s prefill), "down" when below lo by the same margin, otherwise "level".
# (1) A pass is read only if both controls come out as stated above (trk "down" on decode, c24plain "down" on prefill).
# (2) Pass S leaves out every variant whose pass-M decode or prefill is more than 0.5 % below the reference mean: one
#     package serves both design points, so such a variant cannot ship whatever 1 x 255K says. The controls stay in.
# (3) A variant goes to a confirmation pass (n = 3 blocks, shuffled, with the reference, both cells) only if it is "up"
#     on a metric in one pass and "down" on no metric in either pass; only that pass can earn a place in the package.
#     Everything else is reported as measured and dropped.
# (4) If a pass's three reference samples drift monotonically by more than 0.6 % on decode, that pass is repeated.
set -u
W=/root/rocm101-migration-20261114; REF=/root/build-v101-t90; RT=/opt/rocm/core-10.1
P=$W/gpu-opt24-screen.progress
log(){ echo "$(date -Is) [gpuopt24] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
for t in gpuopt24m gpuopt24s; do [ -s $W/screen-$t.tsv ] && { log "FATAL: screen-$t.tsv exists"; exit 1; }; done
declare -A LIB=()
V=()
add(){ # add NAME BUILD
  local n=$1 b=$2 d
  [ -f $b/bin/libggml-hip.so ] || { log "$n: no library, skipped"; return; }
  d=$(python3 -I $W/codegen/devcmp.py $REF $b | grep -oE 'different in [0-9]+' | grep -oE '[0-9]+')
  [ -n "$d" ] || { log "$n: comparison failed, skipped"; return; }
  if [ "$d" = 0 ]; then log "$n: GPU code identical to the reference in every object, not timed"; return; fi
  log "$n: GPU code differs in $d objects"
  LIB[$n]=$b/bin/libggml-hip.so; V+=("$n")
}
for n in ${*:-$(ls -d /root/build-gpuopt24-* 2>/dev/null | sed 's|.*/build-gpuopt24-||')}; do add $n /root/build-gpuopt24-$n; done
[ $# = 0 ] && add v90 /root/build-v101-v90
[ -n "${LIB[trk]:-}" ] || { log "FATAL: the decode control trk is not among the arms"; exit 1; }
[ -f /root/build-master-v101/bin/libggml-hip.so ] || { log "FATAL: no prefill control /root/build-master-v101"; exit 1; }
LIB[c24plain]=/root/build-master-v101/bin/libggml-hip.so
arms(){ # arms NAME ... -> ARMS array: base1, first half, base2, second half, c24plain, base3
  local k=$# h=$(( ($# + 1) / 2 )) i=0 n
  ARMS=("base1=$REF:$RT:$REF/bin/libggml-hip.so")
  for n in "$@"; do
    [ "$i" = "$h" ] && ARMS+=("base2=$REF:$RT:$REF/bin/libggml-hip.so")
    ARMS+=("$n=$REF:$RT:${LIB[$n]}"); i=$((i + 1))
  done
  ARMS+=("c24plain=$REF:$RT:${LIB[c24plain]}" "base3=$REF:$RT:$REF/bin/libggml-hip.so")
}
read_pass(){ # read_pass TSV KEEPFILE -> prints the reading; writes the names that stay for pass S
  python3 -I - "$1" "$2" <<'EOF'
import sys
d = {}
for l in open(sys.argv[1]):
    r = l.rstrip('\n').split('\t')
    try: d[r[1]] = (float(r[3]), float(r[4]))
    except (ValueError, IndexError): print('  %-9s no result: %s' % (r[1] if len(r) > 1 else '?', ' '.join(r[3:])))
base = [d[k] for k in ('base1', 'base2', 'base3') if k in d]
if len(base) < 3: print('fewer than three reference cells: NOT READ'); open(sys.argv[2], 'w').write('NOTREAD\n'); sys.exit(0)
def band(i, floor):
    lo, hi = min(b[i] for b in base), max(b[i] for b in base); return lo, hi, max(hi - lo, floor)
bd, bp = band(0, 0.05), band(1, 1.0)
def cls(v, b): return 'up' if v > b[1] + b[2] else 'down' if v < b[0] - b[2] else 'level'
md = sum(b[0] for b in base) / 3; mp = sum(b[1] for b in base) / 3
print('reference: decode %s (mean %.4f), prefill %s (mean %.1f)' % (' '.join('%.4f' % b[0] for b in base), md, ' '.join('%.1f' % b[1] for b in base), mp))
ok = True
mono = (base[0][0] < base[1][0] < base[2][0]) or (base[0][0] > base[1][0] > base[2][0])
if mono and abs(base[2][0] - base[0][0]) / md > 0.006: print('REFERENCE DRIFTS monotonically by %.3f tok/s: repeat the pass' % (base[2][0] - base[0][0])); ok = False
if 'trk' not in d or cls(d['trk'][0], bd) != 'down': print('CONTROL trk is not down on decode: the pass is NOT READ'); ok = False
if 'c24plain' not in d or cls(d['c24plain'][1], bp) != 'down': print('CONTROL c24plain is not down on prefill: the pass is NOT READ'); ok = False
keep = []
for k, (dec, pre) in sorted(d.items(), key=lambda kv: -kv[1][0]):
    if k.startswith('base'): continue
    cd, cp = cls(dec, bd), cls(pre, bp)
    out = dec / md - 1 < -0.005 or pre / mp - 1 < -0.005
    if k not in ('trk', 'c24plain') and not out: keep.append(k)
    print('  %-9s decode %.4f (%+.2f %%) %-5s  prefill %.1f (%+.2f %%) %-5s%s' % (k, dec, 100 * (dec / md - 1), cd, pre, 100 * (pre / mp - 1), cp, '  [more than 0.5 % down]' if out else ''))
open(sys.argv[2], 'w').write(('\n'.join(keep) + '\n') if ok else 'NOTREAD\n')
EOF
}
k=${#V[@]}; [ "$k" -ge 2 ] || { log "FATAL: nothing to time"; exit 1; }
cd $W || exit 1
arms "${V[@]}"
log "pass M (4 x 64K): $k variants + c24plain + 3 reference cells, about $(( (k + 4) * 64 / 10 )) min"
bash $W/screen.sh gpuopt24m 65536 4 1024 1 "${ARMS[@]}"
log "== pass M reading"; read_pass $W/screen-gpuopt24m.tsv $W/gpu-opt24-keep.txt 2>&1 | tee -a $P
if grep -qx NOTREAD $W/gpu-opt24-keep.txt; then log "pass M is not read: stopping before pass S"; log "=== GPU-OPT24 SCREEN STOPPED ==="; exit 1; fi
S=(); while read -r n; do [ -n "$n" ] && S+=("$n"); done < $W/gpu-opt24-keep.txt
log "pass S keeps ${#S[@]} of $k variants: ${S[*]:-none}"
arms "${S[@]}" trk
log "pass S (1 x 255K): ${#S[@]} variants + trk + c24plain + 3 reference cells, about $(( (${#S[@]} + 5) * 114 / 10 )) min"
bash $W/screen.sh gpuopt24s 260864 1 1024 1 "${ARMS[@]}"
log "== pass S reading"; read_pass $W/screen-gpuopt24s.tsv $W/gpu-opt24-keep-s.txt 2>&1 | tee -a $P
python3 -I - $W/screen-gpuopt24m.tsv $W/screen-gpuopt24s.tsv <<'EOF' 2>&1 | tee -a $P
import sys
def load(p):
    d = {}
    for l in open(p):
        r = l.rstrip('\n').split('\t')
        try: d[r[1]] = (float(r[3]), float(r[4]))
        except (ValueError, IndexError): pass
    return d
def classes(d):
    base = [d[k] for k in ('base1', 'base2', 'base3') if k in d]; out = {}
    if len(base) < 3: return out
    for i, floor in ((0, 0.05), (1, 1.0)):
        lo, hi = min(b[i] for b in base), max(b[i] for b in base); m = max(hi - lo, floor)
        for k, v in d.items():
            if not k.startswith('base'): out.setdefault(k, []).append('up' if v[i] > hi + m else 'down' if v[i] < lo - m else 'level')
    return out
m, s = classes(load(sys.argv[1])), classes(load(sys.argv[2]))
go = [k for k in m if k not in ('trk', 'c24plain') and 'down' not in m[k] + s.get(k, []) and 'up' in m[k] + s.get(k, []) and k in s]
print('to the confirmation pass:', ' '.join(sorted(go)) or 'none')
EOF
log "=== GPU-OPT24 SCREEN DONE ==="
