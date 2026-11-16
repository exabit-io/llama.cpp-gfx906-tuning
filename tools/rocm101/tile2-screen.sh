#!/bin/bash
# tile2-screen.sh [N] -- timing of the second-round tile variants (tile2-build.sh; static properties in
# tile2-build.progress). drv-test.sh settled the first question (2026-11-15: clang 24 compiled directly is as slow as
# through LTO, clang 23 through LTO as fast as directly: the driver pipeline is not the cause). Arms, each the ROCm 10.1
# build on the 10.1 runtime with only the tile object replaced:
#   c23, c24   controls (as drv-test.sh)
#   x23on24    clang 23's optimized bitcode through clang 24's code generator (static: 18 spills, 11 in-loop reloads)
#   x24on23    clang 24's optimized bitcode through clang 23's code generator (static: 1 spill, no in-loop reload)
#   c23none    clang 23 with the compiler's default scheduler instead of max-ilp
#   v90        clang 24 as built + -amdgpu-vgpr-threshold-percent=90 (static: no in-loop reload, 123 VGPRs)
#   st1        clang 24 as built + -amdgpu-use-amdgpu-trackers=1: the scheduler measures register pressure with the
#              AMDGPU trackers instead of the generic ones (default 0 in both compilers). Static: no spill in any of the
#              three kernels, no in-loop reload, and a hot loop whose instruction order is 0.96 similar to rt1's
#   rt1        clang 23 + the same option (static: 1 spill, no in-loop reload; order 0.82 similar to c23's)
# x23on23 is byte-identical to c23 and x24on24 has c24old's hot loop, so the cross method is controlled without timing.
# Cell: 1 x 131072 prefill, -ntg 256, N blocks (default 2).
#
# READING, WRITTEN BEFORE THE FIRST CELL. Statistic: prefill tok/s. "fast" = every sample within 4 tok/s of c23's mean,
# "slow" = every sample within 4 tok/s of c24's mean, otherwise "between". (1) c23 must exceed c24 by more than 8 tok/s
# in every block, or nothing is read. (2) The code generator is the cause if x23on24 is slow and x24on23 is fast; the
# optimizer if x23on24 is fast and x24on23 is slow; anything else is mixed. (3) c23none fast: clang 23 does not need
# max-ilp for this kernel's prefill, so the regression is not specific to the max-ilp schedule; slow or between: it is
# clang 23's max-ilp schedule that clang 24 no longer produces. (4) v90 or st1 fixes the kernel only if fast; then a whole
# ROCm 10.1 build with that option goes to the acceptance A/B at both design points. "between" is reported as partial
# and nothing is built on it. (5) rt1 is read against c23 only: above c23's range by more than 4 tok/s it becomes a
# candidate for the shipping compiler as well (the survey's 'trk' variant measures it on the whole build).
set -u
W=/root/rocm101-migration-20261114; N=${1:-2}
pgrep -x llama-batched-b >/dev/null && { echo "a timing run is in progress" >&2; exit 1; }
ARMS=()
for a in c23 c24 x23on24 x24on23 c23none v90 st1:s-t1 rt1:r-t1; do
  n=${a%%:*}; d=${a##*:}
  [ -f /root/swap/$d/libggml-hip.so.1.0.0 ] || { echo "no library for $n" >&2; exit 1; }
  ARMS+=("$n=/root/build-master-v101:/opt/rocm/core-10.1:/root/swap/$d/libggml-hip.so")
done
cd $W && bash $W/screen.sh tile2 131072 1 256 "$N" "${ARMS[@]}"
