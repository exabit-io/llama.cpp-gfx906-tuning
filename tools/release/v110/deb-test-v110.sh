#!/bin/bash
# deb-test-v110.sh OUT -- the mx-llama-cpp-gfx906 1.1.0 package (last-deb.txt, ROCm 10.1 build), before publishing.
# deb-test-llama-101.sh of the -r2 cycle with this cycle's scripts, plus the DeepSeek-V4.1 checks on the package.
#  1. install the new deb from the file   2. ELF properties: DT_RPATH on its lib dir then on /opt/rocm/core-10.1/lib,
#  every ROCm library from core-10.1, no ggml/llama library from elsewhere   3. the suite on /opt/mx-llama.cpp (no
#  test-backend-ops: the package ships no tests; tbo-pkg-v110.sh does that)   4. MTP row paths at -np 1 and -np 4
#  5. DeepSeek-V4.1 (upstream layout): load, text, perplexity, server, DSpark drafter
# Reference values: the same checks on the source build of the same commit (queue-final.sh, final-*-pr-test).
set -u
S=/root/rebase-v110; O=$1; mkdir -p $O; P=$O/progress
B=/opt/mx-llama.cpp; ROCM=/opt/rocm/core-10.1
DEB=$(cat /root/mx-llama-deb/last-deb.txt)
log(){ echo "$(date -Is) [deb] $*" | tee -a $P; }
[ -f "$DEB" ] || { log "no deb: $DEB"; exit 1; }
log "deb $DEB ($(dpkg-deb -f $DEB Version)); installed now: $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906 2>/dev/null); load $(cut -d' ' -f1-3 /proc/loadavg)"
DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1 apt-get install -y --allow-downgrades "$DEB" > $O/install.out 2>&1 < /dev/null; log "apt-get install rc=$? -> $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906)"
bad=0; n=0
for f in $(dpkg -L mx-llama-cpp-gfx906); do
  [ -f "$f" ] && file -b "$f" | grep -q ELF || continue
  n=$((n+1))
  rp=$(readelf -d "$f" | sed -n 's/.*(RPATH).*\[\(.*\)\]/\1/p')
  case "$rp" in "$B/lib:$ROCM/lib"|"$B/lib:$ROCM/lib:$ROCM/lib") ;; *) bad=$((bad+1)); echo "RPATH: $f ($rp)" >> $O/elf-bad.txt;; esac
done
log "ELF files $n, without DT_RPATH=$B/lib:$ROCM/lib: $bad"
# the worst case for library selection: LD_LIBRARY_PATH names another llama.cpp build AND the other ROCm stream
LDX=/root/build-sub-mxxm/bin:/opt/rocm/core-10.0/lib
log "isolation with LD_LIBRARY_PATH=$LDX: ggml/llama libraries outside $B/lib: $(LD_LIBRARY_PATH=$LDX ldd $B/bin/llama-server | grep -E 'libggml|libllama|libmtmd' | grep -vc "$B/lib"); ROCm libraries outside $ROCM: $(LD_LIBRARY_PATH=$LDX ldd $B/bin/llama-server $B/lib/libggml-hip.so | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | grep -vc "^$ROCM/") of $(LD_LIBRARY_PATH=$LDX ldd $B/lib/libggml-hip.so | awk '$3 ~ /^\/opt\/rocm/' | wc -l)"
log "native: zmm instructions in libggml-cpu $(objdump -d $B/lib/libggml-cpu.so | grep -c zmm), vpdpbusd $(objdump -d $B/lib/libggml-cpu.so | grep -c vpdpbusd)"
log "defaults: $(grep -E 'GGML_HIP_RCCL|FA_QUANTS|MAX_ILP|CMAKE_C_FLAGS:|CMAKE_CXX_FLAGS:|^hip_DIR' /root/mx-llama-deb/build/CMakeCache.txt | grep -v '^//' | tr '\n' ' ' | cut -c1-360)"
log "version: $($B/bin/llama-cli --version 2>&1 | grep version); source $(cat /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT); Depends: $(dpkg-query -W -f='${Depends}' mx-llama-cpp-gfx906 | cut -c1-400)"
SKIP_TBO=1 bash $S/pr-test.sh $B $O/pr-test > $O/pr-test.out 2>&1; log "pr-test deb rc=$?"
log "  allreduce path: $(grep -hiE 'NCCL init failed|probe AllReduce failed' $O/pr-test/*.log | sort | uniq -c | tr '\n' ' ' | cut -c1-200) (empty = RCCL up)"
ARMS="combined host" bash $S/mtp-modes.sh $B $O/mtp-np1 > $O/mtp-np1.out 2>&1; log "deb MTP np1 rc=$?: $(grep 'run [123]' $O/mtp-np1/progress | sed 's/.*gen 256 tok //; s/ | sha.*//' | tr '\n' ';')"
ARMS="combined" bash $S/mtp-modes.sh $B $O/mtp-np4 4 > $O/mtp-np4.out 2>&1; log "deb MTP np4 rc=$?: $(grep 'run [123]' $O/mtp-np4/progress | sed 's/.*gen 256 tok //; s/ | sha.*//' | tr '\n' ';')"
# v1.1.0: DeepSeek-V4.1 in upstream's GGUF layout loads on the package (dsv41-test.sh: -sm layer text, PPL 2K x 4, -sm tensor text,
# server), and the DSpark drafter drafts (ds41-dspark.sh)
RT=$ROCM bash $S/dsv41-test.sh $B $O/dsv41 full > $O/dsv41.out 2>&1; log "dsv41-test deb rc=$?"
grep -v 'clock\|sampler' $O/dsv41/progress 2>/dev/null | cut -c27-330 | while read -r l; do log "  dsv41: $l"; done
RT=$ROCM bash $S/ds41-dspark.sh $B $O/dspark > $O/dspark.out 2>&1; log "ds41-dspark deb rc=$?: $(tail -3 $O/dspark/progress 2>/dev/null | cut -c27-260 | tr '\n' ';')"
log "=== DEB TEST DONE ==="
