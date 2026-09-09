#!/bin/bash
# Rebuilds the b (tile config) and c (k-unroll) ablation builds with the brace fix, sequentially in the ablation worktree.
set -u; W=/root/exabit-llama.cpp-ablation; L=/root/rocm-tests/bench/ablation-rebuild.log
cd $W
build_one() { local name=$1 ref=$2; git checkout -q --detach $ref || { echo "$(date -Is) checkout $ref FAILED" >> $L; return 1; }
  echo "$(date -Is) build $name at $(git rev-parse --short HEAD)" >> $L
  nice -n 19 cmake --build build -j24 --target llama-bench llama-batched-bench llama-perplexity > build-$name-2.log 2>&1 || { echo "$(date -Is) build $name FAILED (build-$name-2.log)" >> $L; return 1; }
  rm -rf /opt/llama.cpp-ablation-$name; cmake --install build --prefix /opt/llama.cpp-ablation-$name > install-$name.log 2>&1 || true
  [ -x /opt/llama.cpp-ablation-$name/bin/llama-bench ] || { mkdir -p /opt/llama.cpp-ablation-$name/bin /opt/llama.cpp-ablation-$name/lib; cp build/bin/llama-bench build/bin/llama-batched-bench build/bin/llama-perplexity /opt/llama.cpp-ablation-$name/bin/; cp build/bin/*.so* /opt/llama.cpp-ablation-$name/lib/; }
  echo "$(date -Is) OK $name" >> $L; }
build_one c 6c2c03a95
# b = the tile config without the k-unroll: c's fix commit rebased onto f48d37902 = f48d37902 + the same one-line fix
git checkout -q --detach f48d37902 && git cherry-pick -q --no-commit 6c2c03a95 2>/dev/null; git -c user.name=Joshua -c user.email=joshua@exabit.io commit -q -am "ablation: brace fix on b" 2>/dev/null; build_one b HEAD
git checkout -q --detach 6c2c03a95; echo "$(date -Is) ABLATION-REBUILT" >> $L
