#!/bin/bash
# confirm.sh -- after the bisect (first bad commit 80df471b75d0, "Reland [MachineLICM] Use
# RegisterClassInfo::getRegPressureSetLimit (#213584)"): is that one line the cause on ROCm 10.1's own compiler commit,
# which llc options move the result there, and does llvm/llvm-project main still have it?
#   B            ROCm 10.1's compiler commit as it is (11 reloads), then with single llc options
#   B-revert     the same commit with MachineLICM's limit read from TargetRegisterInfo again (the line the commit changed)
#   U, U-revert  llvm/llvm-project main as fetched now, as it is and with the same one-line change
# Readings go to steps.log through step.sh. Heavy: never during a timing run.
set -u
T=/root/llvm-bisect; W=/root/rocm101-migration-20261114/llvm-bisect
B=d6f6cb691863e9258034ab48c1dfce21689d5606
export CCACHE_DIR=$T/ccache CCACHE_MAXSIZE=60G
pgrep -x llama-batched-b >/dev/null && { echo "FATAL: a timing run is in progress"; exit 1; }
cd $T/src || exit 1
note(){ echo "$(date -Is) $*" | tee -a $W/steps.log; }
F=llvm/lib/CodeGen/MachineLICM.cpp
revert_line(){ sed -i 's/RegLimit\[i\] = RegClassInfo->getRegPressureSetLimit(i);/RegLimit[i] = TRI->getRegPressureSetLimit(MF, i);/' $F; grep -c 'RegLimit\[i\] = TRI->getRegPressureSetLimit(MF, i);' $F; }
git bisect reset > /dev/null 2>&1
git checkout -q --detach $B || exit 1
note "== confirm: ROCm 10.1's compiler commit $B"
bash $W/step.sh B2
LLCFLAGS=-amdgpu-vgpr-threshold-percent=90 bash $W/step.sh B-t90
LLCFLAGS=-sink-insts-to-avoid-spills bash $W/step.sh B-sink
LLCFLAGS=-disable-machine-licm bash $W/step.sh B-nolicm
LLCFLAGS=-hoist-cheap-insts bash $W/step.sh B-cheap
SCHED= bash $W/step.sh B-defsched
[ "$(revert_line)" = 1 ] || { note "the line to revert was not found at B"; git checkout -q -- $F; exit 1; }
bash $W/step.sh B-revert
SCHED= bash $W/step.sh B-revert-defsched
git checkout -q -- $F
note "== confirm: llvm/llvm-project main"
if git fetch -q --filter=blob:none https://github.com/llvm/llvm-project.git main; then
  U=$(git rev-parse FETCH_HEAD); echo $U > $W/upstream-main.txt
  note "llvm/llvm-project main is $U ($(git log -1 --format='%cd' --date=short $U)); contains 80df471b75d0: $(git merge-base --is-ancestor 80df471b75d0 $U && echo yes || echo no)"
  git checkout -q --detach $U || exit 1
  bash $W/step.sh U
  if [ "$(revert_line)" = 1 ]; then bash $W/step.sh U-revert; else note "the line to revert was not found at main: $(grep -n 'RegLimit\[i\] =' $F | head -2)"; fi
  git checkout -q -- $F
else
  note "fetch of llvm/llvm-project main failed"
fi
note "== confirm done"
