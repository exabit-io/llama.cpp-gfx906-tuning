#!/bin/bash
# night-0909-fasweep.sh: sweep of the gfx906 head-256 FA tile row for ncols=2 (single-stream decode, GQA pair) at 32K depth on one die
# and tp4; ncols 4..32 rows held at the r2 (RDNA2-inherited) values so only the n=1 row moves. Builds first (GPU idle), then measures.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
WT=/root/wt-fa-sweep; TAG=qwen38-27b-fasweep; OUT=$B/$TAG.md
q() { echo "$(date -Is) $*" >> $Q; }
OLD4="256,2,64,128"; OLD8="256,2,64,128"; OLD16="256,2,32,128"; OLD32="256,2,32,128"
# variants for ncols=2: name nthreads,occupancy,nbatch_fa,nbatch_K
VARS="v0:256,2,128,64 v1:128,8,64,64 v2:256,2,64,64 v3:128,4,128,64 v4:256,2,128,128 v5:512,1,128,64 v6:64,8,64,64 v7:128,4,64,128"
q "=== fasweep builds start"
for v in $VARS; do name=${v%%:*}; row=${v#*:}; PFX=/opt/llama.cpp-fa-$name
  [ -x $PFX/bin/llama-bench ] && continue
  ( cd $WT && rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx906 -DAMDGPU_TARGETS=gfx906 \
      -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_CCACHE=ON -DGGML_NATIVE=ON \
      -DBUILD_SHARED_LIBS=ON -DCMAKE_C_FLAGS=-march=native -DCMAKE_CXX_FLAGS=-march=native -DLLAMA_CURL=OFF \
      "-DCMAKE_HIP_FLAGS=-mllvm -amdgpu-sched-strategy=max-ilp -DGFX906_FA2=$row -DGFX906_FA4=$OLD4 -DGFX906_FA8=$OLD8 -DGFX906_FA16=$OLD16 -DGFX906_FA32=$OLD32" \
      -DCMAKE_INSTALL_PREFIX=$PFX > $WT/build-$name.log 2>&1 && cmake --build build -j56 >> $WT/build-$name.log 2>&1 \
      && cmake --install build --prefix $PFX >> $WT/build-$name.log 2>&1 ) && q "fasweep built $name ($row)" || q "fasweep BUILD FAILED $name"
done
q "=== fasweep measure start"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax; start_sampler $B/fasweep-clocks.txt
QUEUE="^/bin/bash $B/night-0909-fasweep[.]sh" DONEFLAG=$B/.fasweep-done SMCLOG=$B/smc-power-fasweep.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-night0909.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-fasweep.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; q "fasweep KILLED"; exit 1; }; trap cleanup INT TERM
echo "# $TAG $(date -Is): gfx906 head-256 tile row for ncols=2 at 32K depth; rows 4..32 = r2 values. rocm0: -p 512 -n 128 -d 32768 -r 2; tp4: -n 128 -d 32768 -r 2" > $OUT
echo "| round | variant | row (threads,occ,nbatch_fa,nbatch_K) | rocm0 pp512@32K | rocm0 tg128@32K | tp4 tg128@32K |" >> $OUT; echo "|---|---|---|---:|---:|---:|" >> $OUT
for r in 1 2; do LIST=$VARS; [ $r = 2 ] && LIST=$(echo $VARS | tr ' ' '\n' | tac | tr '\n' ' ')
  for v in $LIST; do name=${v%%:*}; row=${v#*:}; P=/opt/llama.cpp-fa-$name; [ -x $P/bin/llama-bench ] || continue
    LD_LIBRARY_PATH=$P/lib timeout 1200 $P/bin/llama-bench -m $M --device rocm0 -fa 1 -b 2048 -ub 2048 -p 512 -n 128 -d 32768 -r 2 -o md > $B/$TAG-$name-rocm0-r$r.md 2>/dev/null
    LD_LIBRARY_PATH=$P/lib timeout 1200 $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -b 2048 -ub 2048 -p 0 -n 128 -d 32768 -r 2 -o md > $B/$TAG-$name-tp4-r$r.md 2>/dev/null
    pp=$(grep -E '^\| .*pp512' $B/$TAG-$name-rocm0-r$r.md | awk -F'|' '{gsub(/ /,"",$(NF-1)); print $(NF-1)}'); tg=$(grep -E '^\| .*tg128' $B/$TAG-$name-rocm0-r$r.md | awk -F'|' '{gsub(/ /,"",$(NF-1)); print $(NF-1)}'); t4=$(grep -E '^\| .*tg128' $B/$TAG-$name-tp4-r$r.md | awk -F'|' '{gsub(/ /,"",$(NF-1)); print $(NF-1)}')
    echo "| $r | $name | $row | ${pp:--} | ${tg:--} | ${t4:--} |" >> $OUT; q "fasweep r$r $name ($row): rocm0 pp $pp tg $tg | tp4 tg $t4"
  done; done
{ echo; echo "| variant | row | ppl 16K/6 (r2 = 5.6173; alex row n2 = 5.6418) |"; echo "|---|---|---|"; } >> $OUT
for v in $VARS; do name=${v%%:*}; row=${v#*:}; P=/opt/llama.cpp-fa-$name; [ -x $P/bin/llama-perplexity ] || continue
  LD_LIBRARY_PATH=$P/lib timeout 900 $P/bin/llama-perplexity -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f /root/models/wikitext-2-raw/wiki.test.raw > $B/$TAG-ppl-$name.log 2>&1
  fe=$(grep -oE 'PPL = [0-9.]+ \+/- [0-9.]+' $B/$TAG-ppl-$name.log | tail -1); echo "| $name | $row | ${fe:-FAIL} |" >> $OUT; q "fasweep ppl $name: ${fe:-FAIL}"; done
echo "# done $(date -Is)" >> $OUT; kill $SAMP 2>/dev/null; restore; q "fasweep ALLDONE"; touch $B/.fasweep-done
