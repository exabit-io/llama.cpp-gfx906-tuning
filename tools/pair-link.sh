#!/bin/bash
# How much does the XGMI link matter to a tensor-split pair? Production build, gfx906.env, three pairs on the ring as cabled:
# on-card (0b+0e = rocm0/rocm1, one direct link), bridge (0b+1b = rocm0/rocm2, one direct link), diagonal (0b+1e = rocm0/rocm3, NO direct
# link: two hops or PCIe). llama-bench pp2048/tg128 -r 3 and batched-bench -npl 8 at 2K per pair. The diagonal bounds the link's worth from below.
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; P=/opt/llama.cpp-prod; TAG=qwen38-27b-pair-link
. $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib
OUT=$B/$TAG.md; echo "# $TAG $(date -Is): tp2 on three pairs of the ring as cabled (production build, gfx906.env)" > $OUT
echo "| pair | link | pp2048 | tg128 | batched 8 slots decode | batched 8 prefill |" >> $OUT; echo "|---|---|---:|---:|---:|---:|" >> $OUT
for spec in "on-card:rocm0/rocm1:rocm0,rocm1:one direct XGMI link (0b-0e)" "bridge:rocm0/rocm2:rocm0,rocm2:one direct XGMI link (0b-1b)" "diagonal:rocm0/rocm3:rocm0,rocm3:no direct link (0b-1e: two hops or PCIe)"; do IFS=: read name dev devc desc <<< "$spec"
  timeout 600 $P/bin/llama-bench -m $M --device $dev -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 128 -r 3 -o md > $B/$TAG-$name-lb.md 2>$B/$TAG-$name-lb.err
  lb=$(grep -E '^\| qwen' $B/$TAG-$name-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$(NF-1)); printf "%s | ", $(NF-1)}')
  timeout 600 $P/bin/llama-batched-bench -m $M --device $devc -sm tensor -fa on -b 2048 -ub 2048 -c 34816 -npp 2048 -ntg 128 -npl 8 > $B/$TAG-$name-bb.md 2>/dev/null
  bb=$(grep -E '^\|' $B/$TAG-$name-bb.md | grep -v 'PP \|---' | awk -F'|' '{gsub(/ /,"",$9); gsub(/ /,"",$7); printf "%s | %s |", $9, $7}')
  echo "| $name | $desc | $lb $bb" >> $OUT; echo "$(date -Is) pair-link $name: $lb $bb" >> $B/bench-queue.progress; done
echo "# done $(date -Is)" >> $OUT; restore; for c in 0 1; do echo 413000000 > $RAPL/constraint_${c}_power_limit_uw; done; touch $B/.$TAG-done
