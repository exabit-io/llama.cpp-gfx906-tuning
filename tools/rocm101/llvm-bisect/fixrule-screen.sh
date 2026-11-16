#!/bin/bash
# fixrule-screen.sh [N] -- once the ROCm compiler has upstream's fix (1d7d16b94c61), should the tree's rule for the
# head-256 tile instance (-amdgpu-vgpr-threshold-percent=90) stay? bisect-screen.sh read the fixed compiler WITHOUT the
# rule at 600.5-601.5 tok/s against clang 23's 604.0-604.2 (n = 2), while the rule on the unfixed compiler had read
# 1.1 and 1.6 tok/s ABOVE clang 23 in two earlier sessions (tile2-screen.sh, gpu-opt-confirm.sh). Arms, each the ROCm
# 10.1 build on the 10.1 runtime with only the tile object replaced (llc objects through codegen/llccross.py, clang
# 23's optimized bitcode in every llc arm, llc built at ROCm 10.1's compiler commit d6f6cb691863):
#   c23     control: the object as ROCm 10.0's clang 23 compiles it
#   llcX    + upstream's fix                                    (static: hot loop 1883 / 0 reloads, 1 spill, 128 VGPRs)
#   llcXT   + upstream's fix, -amdgpu-vgpr-threshold-percent=90 (static: 1888 / 0, no spill, 121 VGPRs)
#   llcT    unfixed, -amdgpu-vgpr-threshold-percent=90: what the tree's rule produces today, through llc (1897 / 0)
# Cell: 1 x 131072 prefill, -ntg 256, N blocks (default 3), arm order shuffled in every block.
#
# READING, WRITTEN BEFORE THE FIRST CELL. Statistic: prefill tok/s.
# (1) If llcX is not below c23 in every sample, the earlier 0.5 % was noise: a fixed compiler equals clang 23 on this
#     kernel and (2) is not read.
# (2) The rule ADDS to the fix if every llcXT sample is above every llcX sample; it COSTS if every one is below;
#     otherwise NO DIFFERENCE. ADDS: the rule stays when the compiler is fixed, pending the two-cell check with that
#     compiler. NO DIFFERENCE or COSTS: the rule goes when the compiler has the fix.
# (3) llcT is listed against c23 and llcXT, not classified.
set -u
W=/root/rocm101-migration-20261114; N=${1:-3}
pgrep -x llama-batched-b >/dev/null && { echo "a timing run is in progress" >&2; exit 1; }
ARMS=()
for n in c23 llcX llcXT llcT; do
  [ -f /root/swap/$n/libggml-hip.so.1.0.0 ] || { echo "no library for $n" >&2; exit 1; }
  ARMS+=("$n=/root/build-master-v101:/opt/rocm/core-10.1:/root/swap/$n/libggml-hip.so")
done
cd $W && SHUFFLE=1 bash $W/screen.sh fixrule 131072 1 256 "$N" "${ARMS[@]}"
python3 -I - $W/screen-fixrule.tsv <<'PY'
import collections, sys
d = collections.defaultdict(list)
for l in open(sys.argv[1]):
    r = l.rstrip('\n').split('\t')
    try: d[r[1]].append(float(r[4]))
    except (ValueError, IndexError): print('no result:', ' '.join(r))
for k in ('c23', 'llcX', 'llcXT', 'llcT'):
    v = sorted(d.get(k, []))
    if v: print('[fixrule] %-6s prefill %.1f (%s) n=%d' % (k, sum(v) / len(v), ' '.join('%.1f' % x for x in v), len(v)))
if not all(d.get(k) for k in ('c23', 'llcX', 'llcXT')): print('[fixrule] arms missing: NOT READ'); sys.exit(0)
if not max(d['llcX']) < min(d['c23']): print('[fixrule] (1) llcX is not below c23 in every sample: the fixed compiler equals clang 23 here; (2) not read'); sys.exit(0)
print('[fixrule] (1) llcX below c23 in every sample: %.1f tok/s (%.2f %%)' % (sum(d['llcX']) / len(d['llcX']) - sum(d['c23']) / len(d['c23']), 100 * (sum(d['llcX']) / len(d['llcX']) / (sum(d['c23']) / len(d['c23'])) - 1)))
r = 'ADDS' if min(d['llcXT']) > max(d['llcX']) else 'COSTS' if max(d['llcXT']) < min(d['llcX']) else 'NO DIFFERENCE'
print('[fixrule] (2) the rule on the fixed compiler: %s (%+.1f tok/s)' % (r, sum(d['llcXT']) / len(d['llcXT']) - sum(d['llcX']) / len(d['llcX'])))
PY
