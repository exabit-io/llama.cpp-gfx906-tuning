#!/bin/bash
# rs-bisect.sh -- step 2 of rs-attrib.sh: which of master-v110's commits makes "replay" and "snapshots on the recurrent
# kernel" write different text (-np 4, MTP n_max 3, the three folds off in the environment, unknown to older commits).
# Correctness only (dies at 1000 MHz): no rate is a number. Builds and GPU runs alternate, never overlap.
# good = merge-v1.1.0 (the two arms write the same text), bad = master-v110. A commit is "good" when the two arms give
# the same text on all four requests, "bad" otherwise; a commit that does not build or whose server does not come up
# stops the script (it is not guessed around).
# READING, before the run: the first bad commit names the term. If it is the RCCL-default commit, the cause is the
# exchange and step 1's reading applies. If it is a kernel term, that term computes an operation differently depending
# on something that differs between the two graphs: read it for what, and decide whether that is rounding order (then
# say so, with the operation) or a defect.
set -u
W=/root/rebase-v110; G=/root/exabit-llama.cpp; RT=/opt/rocm/core-10.0; WT=/root/wt-bisect; B=/root/build-bisect; P=$W/rs-bisect.progress; O=$W/rs-bisect
log(){ echo "$(date -Is) [rs-bisect] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is running"; exit 1; }
mkdir -p $O
mapfile -t C < <(git -C $G rev-list --reverse merge-v1.1.0..master-v110)
n=${#C[@]}; log "$n commits; good = merge-v1.1.0, bad = master-v110 ($(git -C $G rev-parse --short master-v110))"
[ -d $WT ] || git -C $G worktree add --detach $WT merge-v1.1.0 > $O/worktree.out 2>&1 || { log "FATAL: worktree"; exit 1; }
lo=-1; hi=$((n - 1))      # C[lo] good (lo = -1: the merge), C[hi] bad
while [ $((hi - lo)) -gt 1 ]; do
  mid=$(( (lo + hi) / 2 )); c=${C[$mid]}; s=$(git -C $G rev-parse --short $c)
  log "step: commit $((mid + 1)) of $n, $s $(git -C $G log -1 --format=%s $c | cut -c1-70)"
  git -C $WT checkout -q --detach $c || { log "FATAL: checkout $s"; exit 1; }
  if ! UIVER=b12774 bash $W/build-rocm.sh $WT $B $O/build-$s.log $RT > $O/build-$s.out 2>&1; then log "FATAL: $s does not build: $(tail -1 $O/build-$s.out)"; exit 1; fi
  GGML_CUDA_NORM_Q8=0 GGML_CUDA_ADD_NORM=0 GGML_CUDA_GDN_PREFUSE=0 RT=$RT ARMS="replay snaprec" bash $W/rs-modes3.sh $B $O/$s > $O/$s.out 2>&1
  t=$(grep -E 'text (long|[0-9])' $O/$s/progress | sed -E 's/.*text ([a-z0-9]+): .*replay ([0-9a-f]+|none) snaprec ([0-9a-f]+|none).*/\1 \2|\3/' | tr '\n' ' ')
  rows=$(grep -cE 'text (long|[0-9])' $O/$s/progress); same=$(grep -E 'text (long|[0-9])' $O/$s/progress | sed -E 's/.*replay ([0-9a-f]+|none) snaprec ([0-9a-f]+|none).*/\1 \2/' | awk '$1 == $2 && $1 != "none"' | wc -l)
  [ "$rows" = 4 ] || { log "FATAL: $s gave $rows text rows: $t"; exit 1; }
  if [ "$same" = 4 ]; then lo=$mid; log "  GOOD ($t); RCCL $(grep -c '^GGML_HIP_RCCL:BOOL=ON' $B/CMakeCache.txt)"; else hi=$mid; log "  BAD ($same of 4 equal: $t); RCCL $(grep -c '^GGML_HIP_RCCL:BOOL=ON' $B/CMakeCache.txt)"; fi
done
log "FIRST BAD: commit $((hi + 1)) of $n: $(git -C $G log -1 --format='%h %s' ${C[$hi]})"
log "=== RS-BISECT DONE ==="
