#!/bin/bash
# accept-v110.sh -- acceptance of the mx-llama-cpp-gfx906 1.1.0 package, and the confirmation that gate8.sh's rule (5)
# asks for before its two defaults replace anything public, in one session.
#
# What changed since the builds that were timed so far (master 9a7177d0a, "a6"): the head-256 tile rule 85 -> 88
# (chosen by mtp-depth-v110b and dp-v110 on exchanged objects), DeepSeek-V4.1 in upstream's layout (a loader),
# MMVQ_MAX_BATCH_SIZE 16 -> 8 and the GGML_TP_AR_MAX_NE default 20481 -> 81921 at four ranks (both chosen by gate8 on
# existing builds, the gate through the environment variable). Here the real builds are timed.
#
# Arms (WANT = the commit the package and s100 must be built from):
#   pkg   the installed package (/opt/mx-llama.cpp: compiled by ROCm 10.1, loads ROCm 10.1 through DT_RPATH), run
#         through a directory of two links (/root/pkg-arm-v110/bin: screen.sh wants BUILD/bin/{llama-batched-bench,
#         libggml-hip.so}); the all-reduce gate is the built-in default (GGML_TP_AR_MAX_NE empty = unset to the code)
#   s100  the same commit by build-rocm.sh with ROCm 10.0 on the 10.0 runtime (/root/build-final-v110); built-in gate
#   old   v1.0.0 master 224b73e24 = the -r2 package's code, by ROCm 10.1 (/root/build-v101-t85): gate 20481, MMVQ 16
#   a6    master before the two defaults, 9a7177d0a, ROCm 10.0 (/root/build-master-v110): gate 20481, MMVQ 16
#   a0    the v1.1.0 merge a3aa1c772, ROCm 10.0 (/root/build-sub-merge): the fork's threshold, MMVQ 8, no Exabit commit
# Cells (Qwen3.8-27B Q8_0, four dies, -sm tensor, f16 KV, 125 W; every block every arm of the cell once in a fresh
# random order; N blocks, default 3), in this order:
#   8 x 4096, 12 x 2048, 16 x 2048  -ntg 512   pkg s100 old a6 a0
#   48 x 512                        -ntg 512   s100 a6 a0
#   4 x 65536, 1 x 260864           -ntg 1024  pkg s100 old a6      (the two design points)
#   8 x 65536                       -ntg 512   pkg old              (the service point: R2.1's 8 slots at R2.2's 64K)
#
# READING, WRITTEN BEFORE THE FIRST RUN. Samples paired by block, mean of ln(a/b), Student's t (df n - 1), Holm over
# the rows of a pair (sub-stats.py). MAJOR PENALTY at a line L = significant after Holm AND the whole 95 % interval
# below -L. Nothing here is compared with another session by number.
# The package:
#   (P1) Design points, pkg against s100: ACCEPTED if no row is a MAJOR PENALTY at the 2 % line (the floor of every
#        package so far, in this cycle's statistic).
#   (P2) Design points, pkg against old: a MAJOR PENALTY row at the 2 % line is a cost of the release against 1.0.0 and
#        goes to the lead BEFORE publishing (rule (3) of dp-v110.sh, which read -0.5 to +0.3 % on the 88 object).
#   (P3) Service point 8 x 65536, pkg against old: the release-over-release numbers; decode per slot is also read
#        against R3.1's 12 tok/s per request and reported either way (1.0.0 is what it is). A MAJOR PENALTY row at
#        the 2 % line goes to the lead before publishing.
# The two defaults (gate8 rule (5)):
#   (C1) The built-in default acts. At 8 x 4096, 12 x 2048 and 16 x 2048 the line "auto-selected two-shot" is in every
#        pkg and s100 log and in no old and no a6 log (the script prints the counts). If not, the code is read before
#        any rate is.
#   (C2) Decode at 8 x 4096, 12 x 2048, 16 x 2048: s100 against a6 a gain with the whole interval above +10 %, +5 %
#        and +2 % (gate8 on existing builds: +14.3 %, +7.0 %, +3.8 %); s100 against a0 no penalty row (gate8: +8.2 %,
#        +0.5 %, +0.4 %). If either fails the package is not published.
#   (C3) Prefill at those three cells, s100 against a6. gate8 read the wider gate 0.5 to 0.9 % behind there, a
#        constant 0.15 to 0.18 s per run; gate9 (B), running while this is written, tests whether that is the
#        benchmark's warm-up batch. Whatever it says, here the difference must again be a constant in seconds:
#        between 0 and 0.3 s per run in each of the three cells (from the cell means). More than 0.3 s in any cell,
#        or a MAJOR PENALTY row at the 2 % line, stops the package until it is understood.
#   (C4) 48 x 512: s100 against a6 no difference expected in either metric (48 rows go to RCCL under both gates, MMQ
#        under both limits); s100 against a0 a gain in decode (gate8: +3.3 % for the old gate). A MAJOR PENALTY row
#        at the 1 % line of s100 against a6 or against a0 stops the package: the unexplained carrier of gate8's rule
#        (4) would then be in what ships.
#   (C5) Design points, s100 against a6: neither default acts at four rows or at one. A MAJOR PENALTY row at the 1 %
#        line stops the package until it is understood; a smaller significant difference is reported with its size.
#   (C6) pkg against old at 8, 12 and 16 slots: the release-over-release gain, for the notes; no rule.
# MTP at depth is the other part of the acceptance: mtp-accept-v110.sh.
# DRY=1: prints the arms, the cells and the estimate, runs nothing. CELLS / N in the environment override.
set -u
W=/root/rebase-v110; TAG=${TAG:-acc}; N=${N:-3}
R100=/opt/rocm/core-10.0; R101=/opt/rocm/core-10.1; PKGD=/opt/mx-llama.cpp; PA=/root/pkg-arm-v110
P=$W/accept-v110.progress
log(){ echo "$(date -Is) [accept] $*" | tee -a $P; }
dir(){ case $1 in pkg) echo $PA;; s100) echo /root/build-final-v110;; old) echo /root/build-v101-t85;; a6) echo /root/build-master-v110;; a0) echo /root/build-sub-merge;; *) echo /nonexistent;; esac; }
rt(){ case $1 in pkg|old) echo $R101;; *) echo $R100;; esac; }
export ARMENV="pkg:GGML_TP_AR_MAX_NE= s100:GGML_TP_AR_MAX_NE="
# slots:depth:ntg:arms
CELLS=${CELLS:-"8:4096:512:pkg,s100,old,a6,a0 12:2048:512:pkg,s100,old,a6,a0 16:2048:512:pkg,s100,old,a6,a0 48:512:512:s100,a6,a0 4:65536:1024:pkg,s100,old,a6 1:260864:1024:pkg,s100,old,a6 8:65536:512:pkg,old"}
sec(){ case $1 in 4:65536) echo 372;; 1:260864) echo 672;; 8:65536) echo 750;; 48:512) echo 103;; *) echo 75;; esac; }
est=0; runs=0
for cell in $CELLS; do arms=${cell##*:}; k=$(echo $arms | tr ',' '\n' | grep -c .); sd=$(echo $cell | cut -d: -f1,2); runs=$(( runs + N * k )); est=$(( est + N * k * $(sec $sd) )); done
if [ "${DRY:-0}" = 1 ]; then echo "cells: $CELLS"; echo "per-arm environment: $ARMENV"; echo "$runs runs, about $(( est / 60 )) min"; exit 0; fi
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is in progress"; exit 1; }
ls $W/screen-$TAG-*.tsv >/dev/null 2>&1 && { log "FATAL: result files of tag $TAG exist"; exit 1; }
WANT=${WANT:?WANT=<commit the package and s100 are built from>}
V=$(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906 2>/dev/null); SC=$(cat /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT 2>/dev/null)
case $V in 1.1.0*.rocm10.1.gfx906.*) ;; *) log "FATAL: the installed package $V is not a 1.1.0 ROCm 10.1 build"; exit 1;; esac
[ "$SC" = "$WANT" ] || { log "FATAL: the installed package is built from $SC, not $WANT"; exit 1; }
[ "$(cat /root/build-final-v110/.src-commit 2>/dev/null)" = "$WANT" ] || { log "FATAL: /root/build-final-v110 is not built from $WANT"; exit 1; }
mkdir -p $PA/bin
ln -sfn $PKGD/bin/llama-batched-bench $PA/bin/llama-batched-bench; ln -sfn "$(readlink -f $PKGD/lib/libggml-hip.so)" $PA/bin/libggml-hip.so
for a in pkg s100 old a6 a0; do
  d=$(dir $a); [ -x $d/bin/llama-batched-bench ] || { log "FATAL: arm $a has no llama-batched-bench in $d/bin"; exit 1; }
  log "arm $a: $d on $(rt $a), source $(cut -c1-9 $d/.src-commit 2>/dev/null) $(cat $d/.src-note 2>/dev/null)$([ $a = pkg ] && echo "package $V, source ${SC:0:9}, GPU library $(sha256sum $PA/bin/libggml-hip.so | cut -c1-16)")"
done
# the package's binary must take its libraries from the package and ROCm 10.1 although screen.sh puts the link
# directory and the arm's ROCm tree on LD_LIBRARY_PATH
out=$(LD_LIBRARY_PATH=$PA/bin:$R101/lib ldd $PA/bin/llama-batched-bench | awk '$2 == "=>" {print $3}' | grep -E 'libggml|libllama|libmtmd' | xargs -r -n1 readlink -f | grep -vc "^$PKGD/lib/")
[ "$out" = 0 ] || { log "FATAL: the package's binary would load $out ggml/llama libraries from outside $PKGD/lib"; exit 1; }
log "per-arm environment: $ARMENV; $N blocks; $runs runs, about $(( est / 60 )) min"
cd $W || exit 1
for cell in $CELLS; do
  S=$(echo $cell | cut -d: -f1); D=$(echo $cell | cut -d: -f2); NTG=$(echo $cell | cut -d: -f3); arms=$(echo $cell | cut -d: -f4)
  ct=s${S}d$D
  SPECS=(); for a in $(echo $arms | tr ',' ' '); do SPECS+=("$a=$(dir $a):$(rt $a)"); done
  SHUFFLE=1 bash $W/screen.sh $TAG-$ct $D $S $NTG "$N" "${SPECS[@]}" > $W/accept-$TAG-$ct.out 2>&1
  rc=$?
  two=""; for a in $(echo $arms | tr ',' ' '); do two="$two $a=$(cat $W/screen-$TAG-$ct/$a-*.log 2>/dev/null | grep -c 'auto-selected two-shot')"; done
  log "cell $ct ($S x $D): rc=$rc, $(grep -c . $W/screen-$TAG-$ct.tsv 2>/dev/null) result rows, $(grep -c 'CELL FAILED\|FA FALLBACK\|RCCL NOT UP' $W/screen-$TAG-$ct.progress 2>/dev/null) failed runs; runs with the two-shot line:$two"
  [ $rc = 0 ] || { log "STOPPED: screen.sh ended with $rc in cell $ct (watchdog or signal); see screen-$TAG-$ct-watchdog.out"; break; }
done
log "--- the package, 2 % line (pairs pkg:s100 pkg:old) ---"
PAIRS="pkg:s100 pkg:old" LINE=2 python3 -I $W/sub-stats.py $TAG 2>&1 | tee -a $P
log "--- the two defaults, 1 % line (pairs s100:a6 s100:a0 pkg:a6) ---"
PAIRS="s100:a6 s100:a0 pkg:a6" LINE=1 python3 -I $W/sub-stats.py $TAG 2>&1 | sed -n '/ against /,$p' | tee -a $P
log "--- (C3) prefill of s100 against a6 in seconds per run, from the cell means ---"
python3 -I - $W $TAG <<'PY' 2>&1 | tee -a $P
import collections, glob, sys
w, tag = sys.argv[1:3]
for path in sorted(glob.glob('%s/screen-%s-*.tsv' % (w, tag))):
    d = collections.defaultdict(list); cell = '?'
    for l in open(path):
        r = l.rstrip('\n').split('\t')
        try: d[r[1]].append(float(r[4])); cell = r[0]
        except (ValueError, IndexError): pass
    if 's100' not in d or 'a6' not in d: continue
    s, dep = [int(x) for x in cell.split(' x ')]
    tok = s * dep
    ms, ma = sum(d['s100']) / len(d['s100']), sum(d['a6']) / len(d['a6'])
    print('  %-11s prefill s100 %.1f, a6 %.1f tok/s over %d tokens: %.2f s against %.2f s, s100 %+.3f s per run' % (cell, ms, ma, tok, tok / ms, tok / ma, tok / ms - tok / ma))
PY
log "=== ACCEPT-V110 $TAG DONE ==="
