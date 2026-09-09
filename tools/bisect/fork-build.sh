#!/bin/bash
# fork-build.sh SHA [jobs]: build the mx-llama.cpp fork at SHA (worktree /root/fork-bisect, same gfx906 recipe, ccache) into /opt/fork-bisect/SHA (llama-bench only).
set -euo pipefail
SHA=$1; J=${2:-20}; W=/root/fork-bisect; P=/opt/fork-bisect/$SHA; mkdir -p /opt/fork-bisect
cd $W; git checkout -q --detach $SHA
cmake -S . -B build-bisect -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx906 -DAMDGPU_TARGETS=gfx906 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_CCACHE=ON -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_C_FLAGS=-march=native -DCMAKE_CXX_FLAGS=-march=native '-DCMAKE_HIP_FLAGS=-mllvm -amdgpu-sched-strategy=max-ilp' -DCMAKE_INSTALL_PREFIX=$P -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF > $P.configure.log 2>&1
nice -n 19 cmake --build build-bisect -j$J --target llama-bench > $P.build.log 2>&1
cmake --install build-bisect --prefix $P > $P.install.log 2>&1 || true
[ -x $P/bin/llama-bench ] || { mkdir -p $P/bin $P/lib; cp build-bisect/bin/llama-bench $P/bin/; cp build-bisect/bin/*.so* $P/lib/ 2>/dev/null || true; }
echo "BUILT $SHA $(date -Is)"
