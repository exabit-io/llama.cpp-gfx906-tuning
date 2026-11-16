#!/bin/bash
# prep.sh -- source and build tree for bisecting the LLVM change that makes the head-256 flash-attention tile kernel spill
# inside its inner loop (tuning report 2026-11-14-rocm-10.1.md section 4.1). The two ROCm compilers name their commits:
#   ROCm 10.0  AMD clang 23.0.0git  ROCm/llvm-project 8f497e0992fb7513f7f78a6f6b6f1056c375e961   (fast kernel)
#   ROCm 10.1  AMD clang 24.0.0git  ROCm/llvm-project d6f6cb691863e9258034ab48c1dfce21689d5606   (slow kernel)
# The loss is made in the code generator (clang 23's optimized bitcode through LLVM 24's code generator is slow), so the
# test needs only llc: it compiles clang 23's bitcode of the tile instance (codegen/irx-x23on23.bc) and counts the
# scratch reloads inside the 16x2 kernel's largest loop (step.sh). Blobless clone: file contents are fetched at checkout.
# Heavy at build time (56 jobs): never while a timing run is in progress.
set -u
T=/root/llvm-bisect; W=/root/rocm101-migration-20261114/llvm-bisect
A=8f497e0992fb7513f7f78a6f6b6f1056c375e961; B=d6f6cb691863e9258034ab48c1dfce21689d5606
pgrep -x llama-batched-b >/dev/null && { echo "FATAL: a timing run is in progress"; exit 1; }
mkdir -p $T
if [ ! -d $T/src/.git ]; then
  git clone --filter=blob:none --no-checkout https://github.com/ROCm/llvm-project.git $T/src || exit 1
fi
cd $T/src || exit 1
for c in $A $B; do git cat-file -e $c^{commit} 2>/dev/null || git fetch --filter=blob:none origin $c || exit 1; done
M=$(git merge-base $A $B) || exit 1
echo "merge base of the two compilers: $M ($(git log -1 --format='%cd %s' --date=short $M | cut -c1-120))"
echo "commits $M..$B: $(git rev-list --count $M..$B); $M..$A: $(git rev-list --count $M..$A)"
echo $M > $W/merge-base.txt
