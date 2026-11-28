#!/bin/bash
# tbo-pkg-v110.sh [OUT] -- test-backend-ops on the GPU code of the INSTALLED 1.1.0 package (the deb ships no tests).
# tbo-r2.sh of the -r2 cycle: the test binary and host libraries of a ROCm 10.1 source build of the SAME commit
# (TB, default /root/build-final-v110-101) with the package's libggml-hip.so preloaded by full path, on the ROCm 10.1
# runtime; one process per die, every op. The loaded GPU library is read back from /proc/<pid>/maps for every
# process; a process that mapped another one is reported as WRONG LIBRARY.
# Passes if all four dies report N/N tests passed with the package's library mapped. (One SOLVE_TRI case near its
# 1e-7 tolerance on one die is the known rare miss of the test: README, "Correctness of master ... final commits".)
set -u
W=/root/rebase-v110; O=${1:-$W/tbo-pkg-v110}; B=${TB:-/root/build-final-v110-101}; RT=/opt/rocm/core-10.1
mkdir -p $O; P=$O/progress
log(){ echo "$(date -Is) [tbo] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
L=$(readlink -f /opt/mx-llama.cpp/lib/libggml-hip.so)
[ -f "$L" ] && [ -x $B/bin/test-backend-ops ] || { log "FATAL: no package library or no test binary"; exit 1; }
log "package $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906), source $(cut -c1-9 /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT); GPU library $L ($(sha256sum "$L" | cut -c1-16)); test binary from $B ($(cut -c1-9 $B/.src-commit 2>/dev/null))"
[ "$(cat /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT)" = "$(cat $B/.src-commit 2>/dev/null)" ] || { log "FATAL: the package and the test binary are not built from the same commit"; exit 1; }
. /root/rocm-tests/bench/gpu-test-env.sh
trap 'restore' EXIT
setmax
PIDS=()
for i in 0 1 2 3; do
  env LD_LIBRARY_PATH=$RT/lib LD_PRELOAD="$L" timeout 5400 $B/bin/test-backend-ops -b ROCm$i > $O/tbo-$i.log 2>&1 &
  PIDS+=($!)
done
sleep 20
for i in 0 1 2 3; do
  c=$(pgrep -P ${PIDS[$i]} | head -1); c=${c:-${PIDS[$i]}}
  awk '$6 ~ /libggml-hip/ {print $6}' /proc/$c/maps 2>/dev/null | sort -u > $O/maps-$i.txt
done
bad=0
for i in 0 1 2 3; do
  wait ${PIDS[$i]}; rc=$?
  m=$(tr '\n' ' ' < $O/maps-$i.txt)
  s=$(grep -E '[0-9]+/[0-9]+ tests passed' $O/tbo-$i.log | tail -1 | sed 's/^ *//')
  [ -n "$s" ] || s="NO SUMMARY"
  [ "$m" = "$L " ] || { m="WRONG LIBRARY: $m"; bad=1; }
  if [[ "$s" =~ ^([0-9]+)/([0-9]+)\ tests\ passed ]] && [ "${BASH_REMATCH[1]}" = "${BASH_REMATCH[2]}" ] && [ "$rc" = 0 ]; then :; else bad=1; fi
  log "ROCm$i rc=$rc: $s; mapped: $m; failed: $(grep -E '^ *[A-Z_0-9]+\(.*FAIL' $O/tbo-$i.log | head -3 | tr '\n' ' ' | cut -c1-300)"
done
log "FLASH_ATTN_EXT cases on ROCm0: $(grep -c '^ *FLASH_ATTN_EXT' $O/tbo-0.log), of them with hsk=256: $(grep -c '^ *FLASH_ATTN_EXT.*hsk=256' $O/tbo-0.log)"
log "verdict: $([ $bad = 0 ] && echo PASSED || echo NOT PASSED)"
log "=== TBO DONE ==="
