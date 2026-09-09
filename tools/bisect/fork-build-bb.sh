#!/bin/bash
# fork-build-bb.sh SHA...: add llama-batched-bench to the existing /opt/fork-bisect/SHA prefixes (same recipe as fork-build.sh, ccache)
set -u; W=/root/fork-bisect; L=/root/rocm-tests/bench/bisect/fork-build-bb.log
for SHA in "$@"; do P=/opt/fork-bisect/$SHA; echo "$(date -Is) start $SHA" >> $L
  cd $W && git checkout -q --detach $SHA || { echo "$(date -Is) CHECKOUT FAIL $SHA" >> $L; continue; }
  cmake -S . -B build-bisect -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx906 -DAMDGPU_TARGETS=gfx906 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_CCACHE=ON -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_C_FLAGS=-march=native -DCMAKE_CXX_FLAGS=-march=native '-DCMAKE_HIP_FLAGS=-mllvm -amdgpu-sched-strategy=max-ilp' -DLLAMA_CURL=OFF > $P.bb-configure.log 2>&1
  nice -n 19 cmake --build build-bisect -j20 --target llama-batched-bench > $P.bb-build.log 2>&1 && cp build-bisect/bin/llama-batched-bench $P/bin/ && cp build-bisect/bin/*.so* $P/lib/ 2>/dev/null; [ -x $P/bin/llama-batched-bench ] && echo "$(date -Is) BUILT $SHA" >> $L || echo "$(date -Is) BUILD FAIL $SHA" >> $L
done; echo "$(date -Is) ALLDONE" >> $L; touch /root/rocm-tests/bench/bisect/.bb-builds-done
