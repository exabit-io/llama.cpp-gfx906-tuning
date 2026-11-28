#!/bin/bash
# deb-test.sh OUT -- the mx-llama-cpp-gfx906 deb named in /root/mx-llama-deb/last-deb.txt, before publishing:
#  1. reference on the package installed now (the previous release): MTP single stream at -np 1 and -np 4
#  2. install the new deb from the local file
#  3. build properties of the installed files: DT_RPATH, library isolation, -march=native code, tree defaults
#  4. suite on /opt/mx-llama.cpp (no test-backend-ops: the package ships no tests), MTP row paths at -np 1 and -np 4
set -u
W=/root/rebase-v110; O=$1; mkdir -p $O; P=$O/progress
B=/opt/mx-llama.cpp
DEB=$(cat /root/mx-llama-deb/last-deb.txt)
log(){ echo "$(date -Is) [deb] $*" | tee -a $P; }
[ -f "$DEB" ] || { log "no deb: $DEB"; exit 1; }
log "deb $DEB ($(dpkg-deb -f $DEB Version)); installed now: $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906 2>/dev/null)"
if [ "$(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906 2>/dev/null)" != "$(dpkg-deb -f $DEB Version)" ]; then
  ARMS="host" bash $W/mtp-modes.sh $B $O/old-np1 > $O/old-np1.out 2>&1; log "previous package MTP np1 rc=$?: $(grep 'run [123]' $O/old-np1/progress | sed 's/.*gen 256 tok //; s/ | sha.*//' | tr '\n' ';')"
  ARMS="host" bash $W/mtp-modes.sh $B $O/old-np4 4 > $O/old-np4.out 2>&1; log "previous package MTP np4 rc=$?: $(grep 'run [123]' $O/old-np4/progress | sed 's/.*gen 256 tok //; s/ | sha.*//' | tr '\n' ';')"
  DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1 apt-get install -y --allow-downgrades "$DEB" > $O/install.out 2>&1 < /dev/null; log "apt-get install rc=$? -> $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906)"
fi
bad=0; n=0
for f in $(dpkg -L mx-llama-cpp-gfx906); do
  [ -f "$f" ] && file -b "$f" | grep -q ELF || continue
  n=$((n+1))
  [ "$(readelf -d "$f" | sed -n 's/.*(RPATH).*\[\(.*\)\]/\1/p')" = "$B/lib" ] || { bad=$((bad+1)); echo "RPATH: $f" >> $O/elf-bad.txt; }
done
log "ELF files $n, without DT_RPATH=$B/lib: $bad"
log "isolation: with LD_LIBRARY_PATH=/root/build-sub-mxxm/bin, libraries outside $B/lib or ROCm: $(LD_LIBRARY_PATH=/root/build-sub-mxxm/bin ldd $B/bin/llama-server | grep -E 'libggml|libllama|libmtmd' | grep -vc "$B/lib")"
log "native: zmm instructions in libggml-cpu $(objdump -d $B/lib/libggml-cpu.so | grep -c zmm), vpdpbusd $(objdump -d $B/lib/libggml-cpu.so | grep -c vpdpbusd)"
log "defaults: $(grep -E 'GGML_HIP_RCCL|FA_QUANTS|MAX_ILP|CMAKE_C_FLAGS:|CMAKE_CXX_FLAGS:' /root/mx-llama-deb/build/CMakeCache.txt | grep -v '^//' | tr '\n' ' ' | cut -c1-300)"
log "version: $($B/bin/llama-cli --version 2>&1 | grep version); source $(cat /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT)"
SKIP_TBO=1 bash $W/pr-test.sh $B $O/pr-test > $O/pr-test.out 2>&1; log "pr-test deb rc=$?"
log "  allreduce path: $(grep -hiE 'NCCL init failed|probe AllReduce failed' $O/pr-test/*.log | sort | uniq -c | tr '\n' ' ' | cut -c1-200) (empty = RCCL up)"
ARMS="combined host" bash $W/mtp-modes.sh $B $O/mtp-np1 > $O/mtp-np1.out 2>&1; log "deb MTP np1 rc=$?"
ARMS="combined" bash $W/mtp-modes.sh $B $O/mtp-np4 4 > $O/mtp-np4.out 2>&1; log "deb MTP np4 rc=$?"
log "=== DEB TEST DONE ==="
