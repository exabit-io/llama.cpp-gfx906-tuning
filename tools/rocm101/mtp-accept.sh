#!/bin/bash
# mtp-accept.sh -- the MTP-at-depth part of the -r2 package's acceptance.
#
# Why: -r2 exists because the -r1 package's tile rule (threshold 90) costs MTP decode at depth (mtp-depth2.sh: -1.84 %
# against no rule; thresholds 85 and 88 +4.4 to +4.9 %). The design-point acceptance (rocm-ab5.sh, llama-batched-bench)
# runs neither cell through the kernel concerned, so the installed package is also read with mtp-depth2.sh's cell:
# llama-server -np 1 -c 131072 --spec-type draft-mtp --spec-draft-n-max 3, a 117.7K-token chat prompt, three requests
# of 768 tokens. All arms run the installed package's llama-server and host libraries on the ROCm 10.1 runtime with
# libggml-hip.so preloaded by full path:
#   pkg   the installed package's own GPU library (must be the -r2 build)
#   r1    /root/build-v101-t90: master 290fc642a by the same compiler = the -r1 package's GPU code (threshold 90)
#   c23   /root/swap/c23: the ROCm 10.1 build's objects with the tile object from ROCm 10.0's clang 23 (listed)
#
# READING, WRITTEN BEFORE THE FIRST CELL. n = N blocks (default 3), every arm once per block in a seeded random order
# (mtp-depth2.sh). "up" / "down": every sample of one arm above / below every sample of the other.
# (1) Decode is compared only between cells with the same generated text and draft counts (mtp-depth2.sh's signature).
# (2) The package passes this part if its MTP decode is "up" against r1 with an effect of at least +3 % (the candidate
#     threshold read +6.3 % in mtp-depth2.sh; half of that is the bar) AND its prefill is not "down" against r1 with an
#     effect below -0.25 %.
# (3) If it does not pass, -r2 is not published and -r1 stays the newest version.
set -u
W=/root/rocm101-migration-20261114; N=${N:-3}; TAG=mtp-accept
P=$W/$TAG.progress
log(){ echo "$(date -Is) [mtpacc] $*" | tee -a $P; }
V=$(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906); SC=$(cat /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT)
case $V in *.rocm10.1.gfx906.*) ;; *) log "FATAL: the installed package $V is not a ROCm 10.1 build"; exit 1;; esac
[ "${SC:0:9}" != 290fc642a ] || { log "FATAL: the installed package is -r1 (290fc642a)"; exit 1; }
PL=$(readlink -f /opt/mx-llama.cpp/lib/libggml-hip.so); R1=$(readlink -f /root/build-v101-t90/bin/libggml-hip.so)
[ -f "$PL" ] && [ -f "$R1" ] || { log "FATAL: a library is missing"; exit 1; }
mkdir -p /root/swap/pkg /root/swap/r1
ln -sfn "$PL" /root/swap/pkg/libggml-hip.so; ln -sfn "$R1" /root/swap/r1/libggml-hip.so
log "package $V (source ${SC:0:9}); pkg = $PL, r1 = $R1; tile rule in the package build: $(grep -oE 'amdgpu-vgpr-threshold-percent=[0-9]+' /root/mx-llama-deb/build/compile_commands.json 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
cd $W || exit 1
TAG=$TAG N=$N ARMS="pkg r1 c23" bash $W/mtp-depth2.sh > $W/$TAG.out 2>&1; log "mtp-depth2.sh rc=$?"
python3 -I - $W/$TAG.tsv "$N" <<'EOF' 2>&1 | tee -a $P
import collections, statistics as st, sys
N = int(sys.argv[2]); d = {}; p = {}; sig = collections.Counter(); rows = []
try: lines = open(sys.argv[1]).read().splitlines()
except OSError as e: print('[mtpacc] no table:', e); sys.exit()
for l in lines:
    a = l.split('\t')
    try: x, y = float(a[3]), float(a[4])
    except ValueError: print('[mtpacc] cell without a result:', a[1], a[2]); continue
    rows.append((a[1], x, y, (a[6], a[7]))); sig[(a[6], a[7])] += 1
if not rows: print('[mtpacc] verdict: NOT PASSED (no cells)'); sys.exit()
good = sig.most_common(1)[0][0]
for arm, x, y, s in rows:
    p.setdefault(arm, []).append(y); d.setdefault(arm, [])
    if s == good: d[arm].append(x)
    else: print('[mtpacc] decode sample of', arm, 'discarded: text or draft counts differ from the majority')
def rel(a, b): return 'up' if min(a) > max(b) else 'down' if max(a) < min(b) else 'level'
for k in ('pkg', 'r1', 'c23'):
    if k in p: print('[mtpacc] %-4s MTP decode n=%d %s; prefill n=%d %.1f (%.1f-%.1f)' % (k, len(d[k]), '%.3f (%.3f-%.3f)' % (st.mean(d[k]), min(d[k]), max(d[k])) if d[k] else 'none', len(p[k]), st.mean(p[k]), min(p[k]), max(p[k])))
ok = all(len(d.get(k, [])) >= N and len(p.get(k, [])) >= N for k in ('pkg', 'r1'))
if ok:
    ed = 100 * (st.mean(d['pkg']) / st.mean(d['r1']) - 1); rd = rel(d['pkg'], d['r1'])
    ep = 100 * (st.mean(p['pkg']) / st.mean(p['r1']) - 1); rp = rel(p['pkg'], p['r1'])
    print('[mtpacc] pkg against r1: MTP decode %+.2f %% %s; prefill %+.2f %% %s' % (ed, rd, ep, rp))
    ok = rd == 'up' and ed >= 3.0 and not (rp == 'down' and ep < -0.25)
else: print('[mtpacc] fewer than N samples in pkg or r1')
print('[mtpacc] verdict: %s' % ('PASSED' if ok else 'NOT PASSED'))
EOF
log "=== MTP ACCEPTANCE DONE ==="
