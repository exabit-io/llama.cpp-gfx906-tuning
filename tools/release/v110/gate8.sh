#!/bin/bash
# gate8.sh -- second stage of attr8.sh: up to which decode width the fork's custom all-reduce pays against RCCL, and up
# to which width MMVQ pays against MMQ, on master-v110.
# Why: attr8 (2026-11-27, n = 3) cut master's 22 Exabit commits into groups and found two carriers of its decode
# penalty against the v1.1.0 merge at 8 and 16 slots (the rest of the groups are gains there):
#   a1, the all-reduce gate group   -12.37 % at 8 x 4096, -2.07 % at 16 x 2048, level at 4 x 4096
#   a4, the MMVQ group               -1.92 % at 16 x 2048 (+1.80 % at 8 x 4096, +0.66 % at 4 x 4096)
# Mechanism of a1, read from that session's logs (the two-shot line is in every a0 log and in no a1..a6 log at 8 and 16
# slots): the merge and mxxm-t's tree have no GGML_TP_AR_MAX_NE at all, so the 20481 that screen.sh exports does
# nothing there, and they send every message under 262144 elements (51 rows of 5120) through the fork's custom
# all-reduce, which since mxxm 19d784ad0 (2026-09-18) switches to its two-shot kernel from 8192 elements at four ranks.
# Master's default 20481 sends everything wider than 4 rows to RCCL. That default was set on 2026-09-08/20 against the
# broadcast kernel alone ("loses at 8-16 rows") and on builds without RCCL, where no threshold applied; neither holds now.
# Mechanism of a4 at 16 columns: of its four commits only term 01 acts there (MMVQ_MAX_BATCH_SIZE 8 -> 16: MMVQ in
# place of MMQ for 9..16 columns); terms 05 and 09 act at one column, term 15 on MUL_MAT_ID.
#
# Arms, every one by the ROCm 10.0 compiler on the 10.0 runtime with RCCL on, as attr8:
#   a0  the merge a3aa1c772 (no gate variable: the fork's threshold)                 /root/build-sub-merge
#   r   master 9a7177d0a, GGML_TP_AR_MAX_NE=20481 (as shipped: RCCL from 5 rows)     /root/build-master-v110
#   c   master 9a7177d0a, GGML_TP_AR_MAX_NE=0 (the fork's threshold, as a0)          /root/build-master-v110
#   m   master with MMVQ_MAX_BATCH_SIZE 8 (gate8-build.sh), GGML_TP_AR_MAX_NE=0      /root/build-attr-m8
# Cells, Qwen3.8-27B Q8_0, four dies, -sm tensor, f16 KV, llama-batched-bench -ntg 512, in this order:
#   8 x 4096 and 16 x 2048 (attr8's cells), 12 x 2048, 24 x 1024, 32 x 1024, 48 x 512: a0 r c (m too at 12 and 16)
#   10 x 2048, 14 x 2048, 15 x 2048: a0 c m
# N blocks (default 3), every block every arm of the cell once in a fresh random order.
#
# READING, WRITTEN BEFORE THE FIRST RUN. Samples paired by block, mean of ln(a/b), Student's t (df n - 1), Holm over the
# rows of a pair (sub-stats.py, LINE=1: MAJOR PENALTY = significant and the whole 95 % interval below -1 %). Nothing
# here is compared with another session by number.
#   (0) Check first. c against r must be the a1 step turned round: decode at 8 x 4096 a gain with the whole interval
#       above +10 %, at 16 x 2048 above +1 %; and the logs must carry the two-shot line in every a0, c and m run and in
#       no r run (the script prints the counts). If not, the gate is not the mechanism: the group's other build default
#       (FA_QUANTS=all) is looked at with a build of its own, and no gate is chosen here.
#   (1) The gate. At each width where r ran, the decode row of c against r says: the custom path PAYS (gain), COSTS
#       (penalty) or is level (no difference shown). The new default lets the custom path take every width up to W*,
#       the widest tested width at which it pays or is level with every narrower tested width paying or level too.
#         W* = 48 (the fork's own threshold ends at 51 rows): the default becomes the fork's threshold wherever RCCL is
#                 built in, and stays 20481 in a build without RCCL (there no threshold applies in the fork's code and
#                 September's -19 % prefill of the ungated path stands).
#         W* < 48: an explicit default of W* rows of 5120 plus one element.
#         W* < 8 cannot happen if (0) holds.
#   (2) Prefill, c against r: expected level in every cell (prefill messages are 2048 rows wide, far over every gate).
#       A MAJOR PENALTY row there stops the change until it is understood; a smaller significant penalty is reported
#       with its size beside the decode gain of the same cell.
#   (3) The MMVQ limit. m against c at 10, 12, 14, 15 and 16 columns, decode row: m ahead (c against m a penalty) says
#       MMVQ costs at that width. The limit becomes L*, the widest tested width at which c is ahead of or level with m,
#       every narrower tested width ahead or level too: 16 = nothing changes (then attr8's a4 step did not repeat and
#       that is said); below 10 = 8, and term 01's widening goes. An untested width takes the verdict of the next
#       narrower tested one.
#   (4) c against a0 is master with the collective made equal: what the Exabit commits are worth per width. A MAJOR
#       PENALTY row that the limit of (3) does not explain names a further carrier, to be looked for before packaging.
#   (5) Nothing on master changes from this run alone: the candidate (gate default and limit together) is built and
#       timed against a0 and a6 at 8 and 16 slots and at the design points before it replaces anything.
# DRY=1: prints the arms, the cells and the estimate, runs nothing.
set -u
W=/root/rebase-v110; RT=/opt/rocm/core-10.0; N=${N:-3}; TAG=${TAG:-gate8}
P=$W/$TAG.progress
log(){ echo "$(date -Is) [gate8] $*" | tee -a $P; }
dir(){ case $1 in a0) echo /root/build-sub-merge;; r|c) echo /root/build-master-v110;; m) echo /root/build-attr-m8;; *) echo /nonexistent;; esac; }
export ARMENV="r:GGML_TP_AR_MAX_NE=20481 c:GGML_TP_AR_MAX_NE=0 m:GGML_TP_AR_MAX_NE=0"
CELLS="8:4096:a0,r,c 16:2048:a0,r,c,m 12:2048:a0,r,c,m 24:1024:a0,r,c 32:1024:a0,r,c 48:512:a0,r,c 10:2048:a0,c,m 14:2048:a0,c,m 15:2048:a0,c,m"
runs=0; for cell in $CELLS; do arms=${cell##*:}; runs=$(( runs + N * $(echo $arms | tr ',' '\n' | grep -c .) )); done
for a in a0 r c m; do
  d=$(dir $a)
  [ -x $d/bin/llama-batched-bench ] && grep -q '^GGML_HIP_RCCL:BOOL=ON' $d/CMakeCache.txt 2>/dev/null || { echo "arm $a: NO BUILD in $d" | tee -a $P; [ "${DRY:-0}" = 1 ] || exit 1; }
done
if [ "${DRY:-0}" = 1 ]; then echo "cells: $CELLS"; echo "per-arm environment: $ARMENV"; echo "$runs runs, about $(( runs * 80 / 60 )) min"; exit 0; fi
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is in progress"; exit 1; }
ls $W/screen-$TAG-*.tsv >/dev/null 2>&1 && { log "FATAL: result files of tag $TAG exist"; exit 1; }
for a in a0 r c m; do d=$(dir $a); log "arm $a: $d, source $(cut -c1-9 $d/.src-commit 2>/dev/null) $(cat $d/.src-note 2>/dev/null), built $(date -r $d/bin/libggml-hip.so +%F\ %H:%M)"; done
log "per-arm environment: $ARMENV; $N blocks per cell; $runs runs, about $(( runs * 80 / 60 )) min"
cd $W || exit 1
for cell in $CELLS; do
  S=${cell%%:*}; rest=${cell#*:}; D=${rest%%:*}; arms=${rest#*:}; ct=w$S
  SPECS=(); for a in $(echo $arms | tr ',' ' '); do SPECS+=("$a=$(dir $a):$RT"); done
  SHUFFLE=1 bash $W/screen.sh $TAG-$ct $D $S 512 "$N" "${SPECS[@]}" > $W/$TAG-$ct.out 2>&1
  rc=$?
  two=""; for a in $(echo $arms | tr ',' ' '); do two="$two $a=$(cat $W/screen-$TAG-$ct/$a-*.log 2>/dev/null | grep -c 'auto-selected two-shot')"; done
  log "cell $ct ($S x $D): rc=$rc, $(grep -c . $W/screen-$TAG-$ct.tsv 2>/dev/null) result rows, $(grep -c 'CELL FAILED\|FA FALLBACK\|RCCL NOT UP' $W/screen-$TAG-$ct.progress 2>/dev/null) failed runs; runs with the two-shot line:$two"
  [ $rc = 0 ] || { log "STOPPED: screen.sh ended with $rc in cell $ct (watchdog or signal); see screen-$TAG-$ct-watchdog.out"; PAIRS="c:r c:m c:a0 r:a0" LINE=1 python3 -I $W/sub-stats.py $TAG >> $P 2>&1; exit 1; }
done
log "--- statistics, LINE=1 (sub-stats.py $TAG; pairs c:r c:m c:a0 r:a0) ---"
PAIRS="c:r c:m c:a0 r:a0" LINE=1 python3 -I $W/sub-stats.py $TAG 2>&1 | tee -a $P
log "=== GATE8 $TAG DONE ==="
