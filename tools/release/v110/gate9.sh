#!/bin/bash
# gate9.sh -- two questions gate8.sh left open before its result becomes commits on master-v110.
# gate8 (2026-11-27, n = 3): the custom all-reduce against RCCL in decode per slot is +14.3 % at 8 rows, +5.7 % at 12,
# +1.9 % at 16 and -2.8 %, -2.1 %, -5.1 % at 24, 32, 48: by its rule the custom path takes messages of up to 16 rows,
# GGML_TP_AR_MAX_NE = 81921 (16 rows of 5120 plus one element). MMQ from 9 columns is ahead of MMVQ at every tested
# width from 10 to 16 (1.0 to 1.8 %): by its rule MMVQ_MAX_BATCH_SIZE goes back to 8.
# The gain is the two-shot kernel's, which the fork selects for small messages at four ranks and more only, so the new
# default is meant for four ranks and more; fewer ranks keep 20481 (no new measurement, broadcast kernel as in September).
#
# (B) Is the prefill penalty of the wider gate a rate or the benchmark's warm-up batch?
#     In gate8 the arm with the custom path open (c) was behind the RCCL-gated arm (r) in PREFILL at 8, 12 and 16 slots
#     (-0.55 % not significant, -0.85 % [-1.17, -0.52], -0.60 % [-0.68, -0.53]) and level at 24, 32, 48. Prefill
#     messages are 2048 rows wide and go to RCCL under every gate, so no prefill message changes path. What does:
#     llama-batched-bench decodes a 16-token warm-up batch before the timed prefill; under 20481 it goes to RCCL, under
#     the fork's threshold to the custom path, and then RCCL's first call falls into the timed prefill. In seconds the
#     three penalties are 0.15, 0.18, 0.17 per run: a fixed cost, not a rate. Test, on ONE build (master + MMVQ limit 8,
#     /root/build-attr-m8) at 12 x 2048, five blocks:
#       r  GGML_TP_AR_MAX_NE=20481  warm-up through RCCL, decode (12 rows) through RCCL
#       k  GGML_TP_AR_MAX_NE=81921  the candidate default: warm-up (16 rows = 81920 elements) custom, decode custom
#       w  GGML_TP_AR_MAX_NE=81920  one element less: warm-up through RCCL, decode custom
#     k and w differ in nothing but the path of the warm-up batch.
# (C) Which default does a build WITHOUT RCCL get? There every message over the gate takes upstream's internal
#     all-reduce, and GGML_TP_AR_MAX_NE=0 leaves no gate at all (the fork's threshold applies with RCCL only).
#     September's 20481 was measured against the broadcast kernel alone. One build (master + MMVQ limit 8,
#     -DGGML_HIP_RCCL=OFF, gate9-build.sh -> /root/build-cand-norccl), three blocks at 8 x 4096, 16 x 2048, 24 x 1024:
#       n20  GGML_TP_AR_MAX_NE=20481   n81  GGML_TP_AR_MAX_NE=81921   n0  GGML_TP_AR_MAX_NE=0 (everything custom)
#     each with GGML_CUDA_ALLREDUCE=internal (what the build falls back to by itself).
# Cells: Qwen3.8-27B Q8_0, four dies, -sm tensor, f16 KV, llama-batched-bench -ntg 512, every block every arm of the
# cell once in a fresh random order; ROCm 10.0 compiler and runtime.
#
# READING, WRITTEN BEFORE THE FIRST RUN. Samples paired by block, mean of ln(a/b), Student's t (df n - 1), Holm over the
# rows of a pair (sub-stats.py, LINE=1). Nothing here is compared with another session by number.
#   (B1) Prefill, 12 x 2048. The warm-up batch is the carrier if w against r shows no difference AND k against w is a
#        penalty. Then the cost is one first call of RCCL per process (about 0.17 s), which this benchmark counts into
#        prefill once its warm-up batch stops making that call; it is reported as that, it is no prefill rate, and it
#        changes nothing about the gate. If w is behind r as k is (k against w no difference): it is NOT the warm-up;
#        the wider gate then carries a prefill penalty of that size (below the 1 % line in gate8), which goes into the
#        commit message beside the decode gain, unexplained. Any other pattern is reported as found, no story fitted.
#   (B2) Decode, 12 x 2048: k against r is gate8's c against r again (a gain, whole interval above +4 %); k against w
#        no difference (both decode through the custom path). If k against r is not that, 81921 is not committed.
#   (C1) Without RCCL the default becomes 81921 as with RCCL if n81 against n20 in DECODE is a gain or no difference at
#        8 x 4096 and at 16 x 2048, and n81 against n20 has no MAJOR PENALTY row in prefill at any of the three widths.
#        A decode penalty at 8 or at 16: a build without RCCL keeps 20481.
#   (C2) 24 x 1024: n81 and n20 send the same messages the same way (24 rows are over both gates); a row that differs
#        there is the warm-up cost of (B) or it is noise, and it is said which by (B)'s result.
#   (C3) n0 becomes nothing. Its rows against n20 and n81 are the record for September's sentence "ungated costs 19 %
#        prefill" on today's code, and for whether a build without RCCL would want the custom path wider than 16 rows
#        (n0 against n81 in decode at 24 x 1024); if that row is a MAJOR gain it is written down as an open item, since
#        no package is built without RCCL.
# DRY=1: prints the arms, the cells and the estimate, runs nothing.
set -u
W=/root/rebase-v110; RT=/opt/rocm/core-10.0; TAG=${TAG:-gate9}
P=$W/$TAG.progress
log(){ echo "$(date -Is) [gate9] $*" | tee -a $P; }
dir(){ case $1 in r|k|w) echo /root/build-attr-m8;; n20|n81|n0) echo /root/build-cand-norccl;; *) echo /nonexistent;; esac; }
rccl(){ case $1 in r|k|w) echo ON;; *) echo OFF;; esac; }
export ARMENV="r:GGML_TP_AR_MAX_NE=20481 k:GGML_TP_AR_MAX_NE=81921 w:GGML_TP_AR_MAX_NE=81920 n20:GGML_CUDA_ALLREDUCE=internal n20:GGML_TP_AR_MAX_NE=20481 n81:GGML_CUDA_ALLREDUCE=internal n81:GGML_TP_AR_MAX_NE=81921 n0:GGML_CUDA_ALLREDUCE=internal n0:GGML_TP_AR_MAX_NE=0"
# slots:depth:arms:blocks
CELLS="12:2048:r,k,w:5 8:4096:n20,n81,n0:3 16:2048:n20,n81,n0:3 24:1024:n20,n81,n0:3"
runs=0; for cell in $CELLS; do n=${cell##*:}; rest=${cell%:*}; arms=${rest##*:}; runs=$(( runs + n * $(echo $arms | tr ',' '\n' | grep -c .) )); done
for a in r k w n20 n81 n0; do
  d=$(dir $a)
  [ -x $d/bin/llama-batched-bench ] && grep -q "^GGML_HIP_RCCL:BOOL=$(rccl $a)" $d/CMakeCache.txt 2>/dev/null || { echo "arm $a: NO BUILD in $d (or RCCL not $(rccl $a))" | tee -a $P; [ "${DRY:-0}" = 1 ] || exit 1; }
done
if [ "${DRY:-0}" = 1 ]; then echo "cells: $CELLS"; echo "per-arm environment: $ARMENV"; echo "$runs runs, about $(( runs * 85 / 60 )) min"; exit 0; fi
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is in progress"; exit 1; }
ls $W/screen-$TAG-*.tsv >/dev/null 2>&1 && { log "FATAL: result files of tag $TAG exist"; exit 1; }
for a in r k w n20 n81 n0; do d=$(dir $a); log "arm $a: $d, source $(cut -c1-9 $d/.src-commit 2>/dev/null) $(cat $d/.src-note 2>/dev/null), built $(date -r $d/bin/libggml-hip.so +%F\ %H:%M)"; done
log "per-arm environment: $ARMENV; $runs runs, about $(( runs * 85 / 60 )) min"
cd $W || exit 1
for cell in $CELLS; do
  n=${cell##*:}; rest=${cell%:*}; S=${rest%%:*}; rest=${rest#*:}; D=${rest%%:*}; arms=${rest#*:}
  first=${arms%%,*}; ct=$(case $first in r) echo b;; *) echo n;; esac)$S
  SPECS=(); for a in $(echo $arms | tr ',' ' '); do SPECS+=("$a=$(dir $a):$RT"); done
  SHUFFLE=1 bash $W/screen.sh $TAG-$ct $D $S 512 "$n" "${SPECS[@]}" > $W/$TAG-$ct.out 2>&1
  rc=$?
  two=""; for a in $(echo $arms | tr ',' ' '); do two="$two $a=$(cat $W/screen-$TAG-$ct/$a-*.log 2>/dev/null | grep -c 'auto-selected two-shot')"; done
  log "cell $ct ($S x $D, $n blocks): rc=$rc, $(grep -c . $W/screen-$TAG-$ct.tsv 2>/dev/null) result rows, $(grep -c 'CELL FAILED\|FA FALLBACK\|RCCL NOT UP' $W/screen-$TAG-$ct.progress 2>/dev/null) failed runs; runs with the two-shot line:$two"
  [ $rc = 0 ] || { log "STOPPED: screen.sh ended with $rc in cell $ct (watchdog or signal); see screen-$TAG-$ct-watchdog.out"; PAIRS="k:r w:r k:w n81:n20 n0:n20 n0:n81" LINE=1 python3 -I $W/sub-stats.py $TAG >> $P 2>&1; exit 1; }
done
log "--- statistics, LINE=1 (sub-stats.py $TAG; pairs k:r w:r k:w n81:n20 n0:n20 n0:n81) ---"
PAIRS="k:r w:r k:w n81:n20 n0:n20 n0:n81" LINE=1 python3 -I $W/sub-stats.py $TAG 2>&1 | tee -a $P
log "=== GATE9 $TAG DONE ==="
