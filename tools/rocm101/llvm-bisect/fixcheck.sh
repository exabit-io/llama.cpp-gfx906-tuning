#!/bin/bash
# fixcheck.sh -- llvm/llvm-project main no longer shows the reloads although it contains the bisected commit. The
# candidate fix is 1d7d16b94c61 "[CodeGen] Avoid register pressure limit underflow (#216372)" (2026-08-19, six days
# after the last upstream commit in ROCm 10.1's compiler, fc636be1b4c2 of 2026-08-13). Readings through step.sh:
#   B-fix        ROCm 10.1's compiler commit + that commit's change to RegisterClassInfo.cpp       (expected good)
#   B-fix-t90    the same + -amdgpu-vgpr-threshold-percent=90 (what the tree's rule does to a fixed compiler)
#   B-print      ROCm 10.1's compiler commit, unchanged except that MachineLICM prints its limits (llc-B-print.log)
#   FXp, FX      upstream at the fix's parent (expected bad) and at the fix (expected good)
# Heavy: never during a timing run.
set -u
T=/root/llvm-bisect; W=/root/rocm101-migration-20261114/llvm-bisect
B=d6f6cb691863e9258034ab48c1dfce21689d5606; FIX=1d7d16b94c614a1ce2ce90bb19edd1e71115f999
export CCACHE_DIR=$T/ccache CCACHE_MAXSIZE=60G
pgrep -x llama-batched-b >/dev/null && { echo "FATAL: a timing run is in progress"; exit 1; }
cd $T/src || exit 1
note(){ echo "$(date -Is) $*" | tee -a $W/steps.log; }
note "== fixcheck: candidate fix $FIX"
git checkout -q --detach $B || exit 1
if git show $FIX -- llvm/lib/CodeGen/RegisterClassInfo.cpp | git apply; then
  bash $W/step.sh B-fix
  LLCFLAGS=-amdgpu-vgpr-threshold-percent=90 bash $W/step.sh B-fix-t90
else
  note "the fix's change to RegisterClassInfo.cpp does not apply to ROCm 10.1's compiler commit"
fi
git checkout -q -- llvm/lib/CodeGen/RegisterClassInfo.cpp
python3 -I - <<'PY'
p = 'llvm/lib/CodeGen/MachineLICM.cpp'; s = open(p).read()
a = '      RegLimit[i] = RegClassInfo->getRegPressureSetLimit(i);\n'
assert s.count(a) == 1
s = s.replace(a, '    {\n      RegLimit[i] = RegClassInfo->getRegPressureSetLimit(i);\n      errs() << "LICMLIMIT " << MF.getName() << " " << TRI->getRegPressureSetName(i) << " tri=" << TRI->getRegPressureSetLimit(MF, i) << " rci=" << RegLimit[i] << "\\n";\n    }\n')
open(p, 'w').write(s)
PY
bash $W/step.sh B-print
note "limits MachineLICM works with in the 16x2 kernel on ROCm 10.1's compiler commit: $(grep 'LICMLIMIT.*flash_attn_tileILi256ELi256ELi16ELi2ELb0' $W/llc-B-print.log | sed 's/LICMLIMIT [^ ]* //' | sort -u | tr '\n' ';' | cut -c1-300)"
note "functions with a limit that differs: $(grep LICMLIMIT $W/llc-B-print.log | awk '{split($4,a,"="); split($5,b,"="); if (a[2] != b[2]) print $2}' | sort -u | wc -l) of $(grep LICMLIMIT $W/llc-B-print.log | awk '{print $2}' | sort -u | wc -l)"
git checkout -q -- llvm/lib/CodeGen/MachineLICM.cpp
git checkout -q --detach $FIX^ || exit 1
bash $W/step.sh FXp
git checkout -q --detach $FIX || exit 1
bash $W/step.sh FX
note "== fixcheck done"
