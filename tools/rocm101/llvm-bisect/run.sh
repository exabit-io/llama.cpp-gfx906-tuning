#!/bin/bash
# run.sh -- validate the endpoints of the bisect with step.sh, then bisect ROCm/llvm-project between the merge base of
# the two ROCm compilers (good) and ROCm 10.1's compiler commit (bad). See prep.sh and step.sh.
#   B  d6f6cb691863  ROCm 10.1's compiler: must read bad (10 reloads in the hot loop when built as clang)
#   A  8f497e0992fb  ROCm 10.0's compiler: must read good (0); A is 97 release-branch commits past the merge base
#   M  merge-base.txt: must read good, or the change is not in M..B and the bisect is not started
# ccache in a directory of its own (the package builds use the default one). Heavy: never during a timing run.
set -u
T=/root/llvm-bisect; W=/root/rocm101-migration-20261114/llvm-bisect
A=8f497e0992fb7513f7f78a6f6b6f1056c375e961; B=d6f6cb691863e9258034ab48c1dfce21689d5606; M=$(cat $W/merge-base.txt)
export CCACHE_DIR=$T/ccache CCACHE_MAXSIZE=60G
pgrep -x llama-batched-b >/dev/null && { echo "FATAL: a timing run is in progress"; exit 1; }
cd $T/src || exit 1
ck(){ git checkout -q --detach "$1" || { echo "checkout of $1 failed" | tee -a $W/steps.log; exit 1; }; }
ck $B; bash $W/step.sh B; rb=$?
ck $A; bash $W/step.sh A; ra=$?
ck $M; bash $W/step.sh M; rm=$?
echo "$(date -Is) endpoints: B rc=$rb (expected 1), A rc=$ra (expected 0), M rc=$rm (expected 0)" | tee -a $W/steps.log
[ $rb = 1 ] && [ $rm = 0 ] || { echo "$(date -Is) endpoints not as expected: bisect not started" | tee -a $W/steps.log; exit 1; }
git bisect reset > /dev/null 2>&1
git bisect start $B $M > $W/bisect.out 2>&1 || { echo "bisect start failed" | tee -a $W/steps.log; exit 1; }
git bisect run bash $W/step.sh >> $W/bisect.out 2>&1
echo "$(date -Is) bisect run rc=$?" | tee -a $W/steps.log
git bisect log > $W/bisect.log 2>&1
grep -B2 -A14 'is the first bad commit' $W/bisect.out | cut -c1-200 | tee -a $W/steps.log
