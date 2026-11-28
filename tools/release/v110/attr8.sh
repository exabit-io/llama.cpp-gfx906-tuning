#!/bin/bash
# attr8.sh -- which of master-v110's Exabit commits carry its decode penalty at 8 and 16 slots.
# Why: the substrate comparison (substrate-v110.sh, 2026-11-27, n = 5) found master against the v1.1.0 merge at
# 8 x 4096: decode -5.25 % [-5.33, -5.17], prefill +2.53 %; at 16 x 2048: decode -3.24 % [-3.28, -3.20], prefill
# +2.52 %; while at 4 x 65536 master is ahead on both (+1.80 %, +1.68 %). The merge against mxxm-t's master is level in
# those cells, so the penalty is ours. -np 8 is the serving recommendation, so it is looked for before v1.1.0 is packaged.
# No earlier measurement separates the commits at these widths (searched: reports, data/raw, the v1.0.0 cycle notes).
#
# Arms: the commits cut into groups along master's history, one build per cut (attr8-build.sh), every one by the
# ROCm 10.0 compiler on the 10.0 runtime with RCCL on, the same environment as the substrate session (screen.sh):
#   a0  merge a3aa1c772                                                    /root/build-sub-merge
#   a1  + RCCL default, AR size gate and its default, FA_QUANTS=all (4)    /root/build-attr-a1
#   a2  + the folds: terms 02 03 04 07 08 10 12 (7)                        /root/build-attr-a2
#   a3  + DPP warp reductions: terms 23 28 (2)                             /root/build-attr-a3
#   a4  + MMVQ: terms 01 15 05 09 (4)                                      /root/build-attr-a4
#   a5  + max-ilp scheduler default (1)                                    /root/build-attr-a5
#   a6  master 9a7177d0a: + LLVM 24 tile rule x3, GDN fold exclusion (4)   /root/build-master-v110
# Cells, Qwen3.8-27B Q8_0, four dies, -sm tensor, f16 KV, llama-batched-bench -ntg 512:
#   m8   8 x 4096      m16  16 x 2048      c4  4 x 4096 (the same depth at 4 columns)
# N blocks (default 3), every block every arm once in a fresh random order.
#
# READING, WRITTEN BEFORE THE FIRST RUN. Samples paired by block, mean of ln(a/b), Student's t (df n - 1), Holm over
# the rows of a pair (sub-stats.py with LINE=1, so its word MAJOR PENALTY here means: significant and the whole 95 %
# interval below -1 %). With n = 3 the test has little power; the substrate session's run-to-run span was 0.1-0.3 %,
# against which a step of 1 % still shows. Nothing here is compared with another session by number.
#   (0) Check first: a6 against a0 must show the penalty again, decode at 8 x 4096 and at 16 x 2048 with the interval
#       wholly below -2 % (LINE=2 reading of the same pair). If it does not, the finding did not repeat: that is the
#       result, and no attribution is read.
#   (1) A step a(k) against a(k-1) whose decode row in a cell is MAJOR PENALTY at LINE=1 names that group a carrier
#       for that cell. The carriers' effects must add up, in log terms, to within 1 point of a6 against a0 in the cell;
#       a larger remainder is spread over steps too small to name, and is reported as that.
#   (2) a6 against a5 is expected level in every row: the four commits between them act only under LLVM 24 (this
#       build is by the ROCm 10.0 compiler) or on GDN ops with replay inputs. A step there is a finding of its own.
#   (3) The 4 x 4096 cell says what a group does at 4 columns at the same depth. A carrier that is level or a gain
#       there can be confined to more than 4 columns without touching what the design points run; one that loses there
#       too is a loser at shallow depth generally, and its gain at the design points comes from depth.
#   (4) An arm whose build failed is absent; the step across it is then the step of two groups together.
# What follows a carrier is a second stage with its own rule, written when the carrier is known: folds -> the three
# environment switches on a6, no build; MMVQ -> its four commits one by one; max-ilp -> the flag per object;
# DPP -> term 23 alone. Nothing on master changes from this run alone: a change is timed against a0 and a6 at 8 and
# 16 slots AND at the design points before it replaces anything.
# DRY=1: prints the arms, their builds and the pairs, runs nothing.
set -u
W=/root/rebase-v110; RT=/opt/rocm/core-10.0; N=${N:-3}; TAG=${TAG:-attr8}
P=$W/$TAG.progress
log(){ echo "$(date -Is) [attr8] $*" | tee -a $P; }
dir(){ case $1 in a0) echo /root/build-sub-merge;; a6) echo /root/build-master-v110;; a[1-5]) echo /root/build-attr-$1;; *) echo /nonexistent;; esac; }
SPECS=(); HAVE=""
for a in a0 a1 a2 a3 a4 a5 a6; do
  d=$(dir $a)
  if [ -x $d/bin/llama-batched-bench ] && grep -q '^GGML_HIP_RCCL:BOOL=ON' $d/CMakeCache.txt 2>/dev/null; then
    SPECS+=("$a=$d:$RT"); HAVE="$HAVE $a"
  else
    [ "${DRY:-0}" = 1 ] && echo "arm $a: no build in $d yet" || log "arm $a: NO BUILD in $d, left out"
  fi
done
PAIRS=""; prev=""
for a in $HAVE; do [ -n "$prev" ] && PAIRS="$PAIRS $a:$prev"; prev=$a; done
case "$PAIRS" in *" a6:a0"*) ;; *) PAIRS="$PAIRS a6:a0";; esac
if [ "${DRY:-0}" = 1 ]; then echo "arms:$HAVE"; printf '  %s\n' "${SPECS[@]}"; echo "pairs:$PAIRS"; echo "about $(( ${#SPECS[@]} * N * 6 )) min"; exit 0; fi
case " $HAVE " in *" a0 "*" a6 "*) ;; *) log "FATAL: a0 or a6 has no build"; exit 1;; esac
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is in progress"; exit 1; }
ls $W/screen-$TAG-*.tsv >/dev/null 2>&1 && { log "FATAL: result files of tag $TAG exist"; exit 1; }
for s in "${SPECS[@]}"; do a=${s%%=*}; d=$(dir $a); log "arm $a: $d, source $(cut -c1-9 $d/.src-commit 2>/dev/null), built $(date -r $d/bin/libggml-hip.so +%F\ %H:%M)"; done
log "arms:$HAVE; $N blocks per cell; about $(( ${#SPECS[@]} * N * 6 )) min"
cd $W || exit 1
bench(){ # bench CELLTAG D S NTG
  SHUFFLE=1 bash $W/screen.sh $TAG-$1 $2 $3 $4 "$N" "${SPECS[@]}" > $W/$TAG-$1.out 2>&1
  local rc=$?
  log "cell $1 ($3 x $2): rc=$rc, $(grep -c . $W/screen-$TAG-$1.tsv 2>/dev/null) result rows, $(grep -c 'CELL FAILED\|FA FALLBACK\|RCCL NOT UP' $W/screen-$TAG-$1.progress 2>/dev/null) failed runs"
  [ $rc = 0 ] || { log "STOPPED: screen.sh ended with $rc in cell $1 (watchdog or signal); see screen-$TAG-$1-watchdog.out"; PAIRS="$PAIRS" LINE=1 python3 -I $W/sub-stats.py $TAG >> $P 2>&1; exit 1; }
}
bench m8 4096 8 512
bench m16 2048 16 512
bench c4 4096 4 512
log "--- steps, LINE=1 (sub-stats.py $TAG; pairs:$PAIRS) ---"
PAIRS="$PAIRS" LINE=1 python3 -I $W/sub-stats.py $TAG 2>&1 | tee -a $P
log "--- check (0), a6 against a0 at LINE=2 ---"
PAIRS="a6:a0" LINE=2 python3 -I $W/sub-stats.py $TAG 2>&1 | sed -n '/against/,$p' | tee -a $P
log "=== ATTR8 $TAG DONE ==="
