#!/bin/bash
# bisect-screen.sh [N] -- does the commit the bisect found (80df471b75d0, "Reland [MachineLICM] Use
# RegisterClassInfo::getRegPressureSetLimit (#213584)") carry the prefill loss, not only the reloads? step.sh counts
# reloads in the 16x2 kernel's hot loop; this times the same objects. Arms, each the ROCm 10.1 build on the 10.1 runtime
# with only the head-256 tile object replaced (codegen/llccross.py for the llc objects; clang 23's optimized bitcode in
# every llc arm):
#   c23, c24   controls: the object as ROCm 10.0's clang 23 and as ROCm 10.1's clang 24 compile it (tile2-screen.sh's)
#   llcP       llc built at 2fe9fc90780d, the commit's parent            (static: 0 reloads, spills 5 / 4 / 1)
#   llcF       llc built at 80df471b75d0, the first bad commit           (static: 11 reloads, spills 10 / 14 / 18)
#   llcR       llc built at ROCm 10.1's compiler commit d6f6cb691863 with MachineLICM reading its limit from
#              TargetRegisterInfo again (the one line the commit changed)
#   llcX       llc built at ROCm 10.1's compiler commit with upstream's later fix applied instead (1d7d16b94c61
#              "[CodeGen] Avoid register pressure limit underflow (#216372)", its change to RegisterClassInfo.cpp)
# llc at ROCm 10.0's compiler commit reproduces c23's machine code byte for byte, so the method needs no timed control.
# Cell: 1 x 131072 prefill, -ntg 256, N blocks (default 2), as tile2-screen.sh; the arm order is shuffled in every block.
#
# READING, WRITTEN BEFORE THE FIRST CELL. Statistic: prefill tok/s. "fast" = every sample within 4 tok/s of c23's mean,
# "slow" = every sample within 4 tok/s of c24's mean, otherwise "between".
# (1) c23 must exceed c24 by more than 8 tok/s in every block, or nothing is read.
# (2) The commit carries the loss if llcP is fast and llcF is slow. Anything else: the bisect's static criterion found a
#     commit that changes the reloads, and the timing does not confirm that it is the cause of the loss.
# (3) The one line carries it on ROCm 10.1's compiler if llcR is fast; "between" means other changes after the commit
#     contribute; slow means the line alone does not restore the kernel there.
# (4) Upstream's fix restores the kernel on ROCm 10.1's compiler if llcX is fast (same classes as (3)).
set -u
W=/root/rocm101-migration-20261114; N=${1:-2}
pgrep -x llama-batched-b >/dev/null && { echo "a timing run is in progress" >&2; exit 1; }
ARMS=()
for n in c23 c24 llcP llcF llcR llcX; do
  [ -f /root/swap/$n/libggml-hip.so.1.0.0 ] || { echo "no library for $n" >&2; exit 1; }
  ARMS+=("$n=/root/build-master-v101:/opt/rocm/core-10.1:/root/swap/$n/libggml-hip.so")
done
cd $W && SHUFFLE=1 bash $W/screen.sh bisect 131072 1 256 "$N" "${ARMS[@]}"
python3 -I - $W/screen-bisect.tsv <<'EOF'
import collections, sys
d = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    r = l.rstrip('\n').split('\t')
    try: d[r[1]][r[2]] = float(r[4])
    except (ValueError, IndexError): print('no result:', ' '.join(r))
m = {k: sum(v.values()) / len(v) for k, v in d.items() if v}
if 'c23' not in m or 'c24' not in m: print('controls missing: NOT READ'); sys.exit(0)
ok = all(d['c23'].get(b, 0) - d['c24'].get(b, 1e9) > 8 for b in d['c23'])
print('[bisect] controls: c23 %.1f, c24 %.1f, apart by more than 8 tok/s in every block: %s' % (m['c23'], m['c24'], ok))
cls = {}
for k, v in d.items():
    s = sorted(v.values())
    cls[k] = 'fast' if all(abs(x - m['c23']) <= 4 for x in s) else 'slow' if all(abs(x - m['c24']) <= 4 for x in s) else 'between'
    print('[bisect] %-5s prefill %.1f (%s) n=%d: %s' % (k, m[k], ' '.join('%.1f' % x for x in s), len(s), cls[k]))
if not ok: print('[bisect] rule (1) fails: nothing is read'); sys.exit(0)
print('[bisect] the commit carries the loss: %s' % ('YES' if cls.get('llcP') == 'fast' and cls.get('llcF') == 'slow' else 'NOT CONFIRMED'))
print('[bisect] the one line restores the kernel on ROCm 10.1\'s compiler commit: %s' % {'fast': 'YES', 'between': 'PARTLY', 'slow': 'NO'}.get(cls.get('llcR'), '?'))
print('[bisect] upstream\'s fix 1d7d16b94c61 restores the kernel on ROCm 10.1\'s compiler commit: %s' % {'fast': 'YES', 'between': 'PARTLY', 'slow': 'NO'}.get(cls.get('llcX'), '?'))
EOF
