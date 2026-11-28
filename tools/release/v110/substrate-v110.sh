#!/bin/bash
# substrate-v110.sh -- what following upstream llama.cpp's releases costs or gains against mxxm-t's fork on this box.
# Why (lead, 2026-11-26): none of the merges offered to mxxm-t was taken, so the two lines diverge. Our fork is its own
# project: it follows upstream and keeps the gfx906 patches (his and ours), provided that following upstream carries no
# MAJOR, STATISTICALLY SIGNIFICANT performance penalty against his fork. Dropping his patches is NOT a candidate.
# No measurement of this kind exists (the last stock-against-fork numbers are September's), so this is new work.
#
# Arms, every one by the ROCm 10.0 compiler on the 10.0 runtime, the tree's own defaults plus RCCL on (sub-build.sh),
# the same environment for all:
#   mxxm    mxxm-t's master 557a6068f (his merge of b12500 + his 202 commits)     /root/build-sub-mxxm
#   merge   his master + upstream v1.1.0 + three fixes (a3aa1c772)                /root/build-sub-merge
#   master  merge + the 22 Exabit commits (9a7177d0a)                             /root/build-master-v110
#   stock   OPTIONAL (ARMS="stock mxxm merge master", WITH_STOCK=1 for sub-build.sh): upstream v1.1.0 unmodified,
#           /root/build-sub-stock. Not a candidate; it only says what the patches are worth over upstream today.
# Cells, Qwen3.8-27B Q8_0 on four dies, -sm tensor, f16 KV, in this order:
#   m4   llama-batched-bench 4 x 65536, -ntg 1024     the multi-user design point
#   mtp  llama-server, 1 stream, draft-mtp n_max 3, about 47K tokens of prompt, 5 requests of 512 tokens (sub-mtp.sh)
#   m8   llama-batched-bench 8 x 4096, -ntg 512       the serving recommendation (-np 8)
#   m16  llama-batched-bench 16 x 2048, -ntg 512      the small-batch MMQ shape
#   s1   llama-batched-bench 1 x 260864, -ntg 1024    the single-user design point
# N blocks per cell (default 5); every block runs every arm once in a fresh random order (screen.sh SHUFFLE=1).
#
# READING, WRITTEN BEFORE THE FIRST RUN (sub-stats.py applies it; nothing here is compared with another session).
# Statistic: the samples of two arms are paired by block; mean of ln(a_i / b_i), Student's t with n - 1 degrees of
# freedom, 95 % interval of the effect, Holm's correction over all rows (cell x metric) of one pair.
#   significant = adjusted p < 0.05
#   MAJOR       = significant AND the whole 95 % interval beyond 2 % (the acceptance floor of every package so far;
#                 LINE=<percent> in the environment of sub-stats.py reads the same data against another line)
# The pairs and the question each answers:
#   (1) merge  against mxxm   the same patches on upstream v1.1.0 and on b12500: what FOLLOWING UPSTREAM'S RELEASES
#                             does, with nothing else changed. This is the lead's condition.
#   (2) master against mxxm   what we ship against his fork as it stands
#   (3) master against merge  the Exabit commits
#   With the optional arm: stock against mxxm / merge and master against stock = what the patches are worth.
# The condition is MET when (1) has no MAJOR PENALTY row. A MAJOR PENALTY row in (1) that (2) does not have says the
# release costs something our commits make up for; one in both is reported to the lead with its size and its cell
# before anything is built on v1.1.0, and the cause is looked for in upstream's 274 commits since b12500.
# Further rules:
#   (a) An arm that cannot run a cell at all (abort, out of memory, no MTP) is a finding of its own; its rows are absent.
#       A row with fewer than 3 pairs has no verdict.
#   (b) Every arm must exchange the same way: the custom allreduce for messages under GGML_TP_AR_MAX_NE elements, RCCL
#       for the rest. A working RCCL logs nothing; what shows is its three failure warnings, and screen.sh / sub-mtp.sh
#       discard a run that has one. Checked once per build by sub-exchange.sh (2026-11-27: the three arms link
#       librccl, log the same custom-allreduce path and no warning). The environment gives EVERY arm the 20481-element
#       gate, which is master's default and not his (262144): his arm runs with our tuned gate, i.e. better at prefill
#       than out of the box. That is deliberate: the arms differ in code only.
#       CORRECTION 2026-11-27 (after the run, found by attr8.sh): the sentence above is wrong. mxxm-t's tree and the
#       merge have no GGML_TP_AR_MAX_NE; the variable does nothing there and both ran the fork's own threshold (custom
#       path under 262144 elements = up to 51 rows). Pair (1) compared like with like (both arms on that threshold); at
#       1 and 4 slots and in the MTP cell every arm is on the custom path at decode and on RCCL at prefill, so nothing
#       changes there. At 8 and 16 slots master (RCCL from 5 rows) and the other two arms differed in the collective:
#       that IS the decode penalty of pairs (2) and (3) there. See README.md, section on attr8.
#   (c) MTP decode: the trees write different text on the same prompts (upstream's numerics moved between b12500 and
#       v1.1.0), so the acceptance differs by text; the row is read with the accepted/drafted counts printed beside it,
#       and a decode difference that the acceptance counts explain is said to be that.
#   (d) Clock: every watchdog log's clock column is checked (substrate-v110.sh stops at the first cell a watchdog stops).
# TRY=1: a functional pass (small cells, one block, clock guard off, tag "subtry"); none of its rates is a number.
set -u
W=/root/rebase-v110; RT=/opt/rocm/core-10.0; N=${N:-5}; TAG=${TAG:-sub}; ARMS=${ARMS:-"mxxm merge master"}
if [ "${TRY:-0}" = 1 ]; then TAG=subtry; N=1; export SCLK_GUARD=0 CHARS=20000 NPRED=128; fi
P=$W/$TAG.progress
log(){ echo "$(date -Is) [substrate] $*" | tee -a $P; }
dir(){ case $1 in stock) echo /root/build-sub-stock;; mxxm) echo /root/build-sub-mxxm;; merge) echo /root/build-sub-merge;; master) echo /root/build-master-v110;; *) echo /nonexistent;; esac; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is in progress"; exit 1; }
ls $W/screen-$TAG-*.tsv $W/$TAG-mtp.tsv >/dev/null 2>&1 && { log "FATAL: result files of tag $TAG exist"; exit 1; }
SPECS=(); MSPECS=()
for a in $ARMS; do
  d=$(dir $a)
  [ -x $d/bin/llama-batched-bench ] && [ -x $d/bin/llama-server ] || { log "FATAL: arm $a has no build in $d (sub-build.sh)"; exit 1; }
  grep -q '^GGML_HIP_RCCL:BOOL=ON' $d/CMakeCache.txt || { log "FATAL: arm $a is built without RCCL"; exit 1; }
  SPECS+=("$a=$d:$RT"); MSPECS+=("$a=$d")
  log "arm $a: $d, source $(cat $d/.src-commit 2>/dev/null | cut -c1-9), built $(date -r $d/bin/libggml-hip.so +%F\ %H:%M)"
done
k=${#SPECS[@]}
log "arms: $ARMS; $N blocks per cell; about $(( k * N * 25 )) min at full clock"
cd $W || exit 1
arline(){ # arline CELLTAG -> the allreduce line of each arm's first log
  local a f
  for a in $ARMS; do
    f=$(ls $W/screen-$1/$a-*.log 2>/dev/null | head -1)
    log "  allreduce, $a: $([ -n "$f" ] && grep -h -i -m2 -E 'allreduce|nccl|rccl' "$f" | tr '\n' ' ' | cut -c1-220)"
  done
}
bench(){ # bench CELLTAG D S NTG
  SHUFFLE=1 bash $W/screen.sh $TAG-$1 $2 $3 $4 "$N" "${SPECS[@]}" > $W/$TAG-$1.out 2>&1
  local rc=$?
  log "cell $1 ($3 x $2): rc=$rc, $(grep -c . $W/screen-$TAG-$1.tsv 2>/dev/null) result rows, $(grep -c 'CELL FAILED\|FA FALLBACK\|RCCL NOT UP' $W/screen-$TAG-$1.progress 2>/dev/null) failed runs"
  arline $TAG-$1
  [ $rc = 0 ] || { log "STOPPED: screen.sh ended with $rc in cell $1 (watchdog or signal); see screen-$TAG-$1-watchdog.out"; python3 -I $W/sub-stats.py $TAG >> $P 2>&1; exit 1; }
}
if [ "${TRY:-0}" = 1 ]; then
  bench m4 2048 4 64
  bash $W/sub-mtp.sh $TAG 1 "${MSPECS[@]}" > $W/$TAG-mtp.out 2>&1; log "cell mtp: rc=$?, $(grep -c . $W/$TAG-mtp.tsv 2>/dev/null) rows"
  bench m8 2048 8 64
  bench m16 2048 16 64
  bench s1 4096 1 64
else
  bench m4 65536 4 1024
  bash $W/sub-mtp.sh $TAG "$N" "${MSPECS[@]}" > $W/$TAG-mtp.out 2>&1
  rc=$?; log "cell mtp: rc=$rc, $(grep -c . $W/$TAG-mtp.tsv 2>/dev/null) rows"
  [ $rc = 0 ] || { log "STOPPED: sub-mtp.sh ended with $rc; see $TAG-mtp-watchdog.out"; python3 -I $W/sub-stats.py $TAG >> $P 2>&1; exit 1; }
  bench m8 4096 8 512
  bench m16 2048 16 512
  bench s1 260864 1 1024
fi
log "--- statistics (sub-stats.py $TAG) ---"
python3 -I $W/sub-stats.py $TAG 2>&1 | tee -a $P
log "=== SUBSTRATE $TAG DONE ==="
