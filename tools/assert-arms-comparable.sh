#!/bin/bash
# assert-arms-comparable.sh BUILD_A BUILD_B [BUILD_C ...]
#
# Refuse to spend GPU time on arms whose builds differ in anything except what is under test.
#
# WHY THIS EXISTS: on 2026-09-22 a 14-cell screen (1.6 GPU hours) was run with delta-minus builds that
# carried the default GGML_CUDA_FA_QUANTS while the comparison arm carried the campaign value. All six
# minus-builds came in 2.4-2.7% below the bundle -- IDENTICALLY. Six unrelated commits cannot each cost
# the same 2.5%; that was the build difference, and it is above the 2% materiality floor. Roughly 6 GPU
# hours across the campaign have been discarded for this one mistake shape: measuring before checking
# that the arms were comparable.
#
# This check costs seconds. It runs BEFORE the first cell, and it fails loudly.
set -u
[ $# -ge 2 ] || { echo "usage: assert-arms-comparable.sh BUILD_A BUILD_B [...]" >&2; exit 2; }
# Every option that changes generated code or the compiled kernel set. If two arms differ here, the
# contrast between them is not the unit under test.
KEYS="GGML_HIP_RCCL GGML_CUDA_FA_QUANTS GGML_CUDA_FA GGML_CUDA_FA_ALL_QUANTS GGML_HIP_GRAPHS
      GGML_HIP_NO_VMM GGML_HIP_MMQ_MFMA GGML_CUDA_FORCE_MMQ GGML_CUDA_FORCE_CUBLAS GGML_NATIVE
      GGML_CUDA_NO_PEER_COPY CMAKE_BUILD_TYPE AMDGPU_TARGETS CMAKE_HIP_ARCHITECTURES"
fail=0
declare -A seen
for bd in "$@"; do
  [ -f "$bd/CMakeCache.txt" ] || { echo "FAIL  $bd has no CMakeCache.txt"; fail=1; continue; }
  for k in $KEYS; do
    v=$(grep -E "^$k:(BOOL|STRING|FILEPATH|PATH)=" "$bd/CMakeCache.txt" | head -1 | cut -d= -f2-)
    key="$k"
    if [ -z "${seen[$key]+x}" ]; then seen[$key]="$v|$bd"
    else
      prev="${seen[$key]%%|*}"; pbd="${seen[$key]#*|}"
      if [ "$v" != "$prev" ]; then
        echo "FAIL  $k differs:"
        echo "        $(basename $pbd): '${prev:-<unset>}'"
        echo "        $(basename $bd): '${v:-<unset>}'"
        fail=1
      fi
    fi
  done
done
if [ $fail -eq 0 ]; then
  echo "OK    ${#@} arms share every code-affecting build option"
else
  echo "REFUSING: the arms are not comparable. Fix the builds before spending GPU time."
fi
exit $fail
