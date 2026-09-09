#!/bin/bash
# S6 batch-1 MMVQ variants on top of the fusion tree: each sets the 1-column knobs, rebuilds mmvq.cu, installs to /opt/llama.cpp-nq-b1-<tag>,
# then the defaults are restored and /opt/llama.cpp-mxxm-fh-nq re-installed. CPU only, nice 19.
set -u
W=/root/mx-llama.cpp-fusion; F=$W/ggml/src/ggml-cuda/mmvq.cu; LOG=/root/rocm-tests/bench/b1-variants-build.log; : > $LOG
log() { echo "$(date -Is) $*" >> $LOG; }
setk() { sed -i "s/^#define GGML_MMVQ_GCN_ROWS1 .*/#define GGML_MMVQ_GCN_ROWS1 $1/; s/^#define GGML_MMVQ_GCN_NWARPS1 .*/#define GGML_MMVQ_GCN_NWARPS1 $2/; s/^#define GGML_MMVQ_GCN_Q8_VDR8_1COL .*/#define GGML_MMVQ_GCN_Q8_VDR8_1COL $3/" $F; }
build() { local tag=$1; shift; setk "$@"; log "$tag: knobs ROWS1=$1 NWARPS1=$2 VDR8_1COL=$3"
  t0=$(date +%s); nice -n 19 cmake --build $W/build -j16 > $W/build-b1-$tag.log 2>&1 || { log "$tag: BUILD FAILED $(grep -m2 error: $W/build-b1-$tag.log | cut -c1-150)"; return 1; }
  cmake --install $W/build --prefix /opt/llama.cpp-nq-b1-$tag > /dev/null 2>&1 && log "$tag: installed in $(( $(date +%s) - t0 )) s"; }
build r2   2 2 0
build r4   4 2 0
build v8   1 2 1
build r2v8 2 2 1
build w4   1 4 0
setk 1 2 0; nice -n 19 cmake --build $W/build -j16 > $W/build-b1-restore.log 2>&1 && cmake --install $W/build --prefix /opt/llama.cpp-mxxm-fh-nq > /dev/null 2>&1 && log "defaults restored, /opt/llama.cpp-mxxm-fh-nq re-installed"
git -C $W diff --stat | tail -1 >> $LOG; touch /root/rocm-tests/bench/.b1-builds-done; log DONE
