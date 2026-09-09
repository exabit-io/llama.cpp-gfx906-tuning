#!/bin/bash
# After post24: the branch's 8-column decode cell reads 6-9% under production with --no-repack (163.6 vs 175.0). Is it the fork's
# mat-vec fusion routing around the gfx906 Q8_0 fast path (q8_fast requires !has_fusion)? Batched 8/12/16 with GGML_CUDA_DISABLE_FUSION=1 and 0.
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; RAPL=/sys/class/powercap/intel-rapl:0; Q=$B/bench-queue.progress
q() { echo "$(date -Is) $*" >> $Q; }
q "=== post25b start (8-column fusion check)"; . $B/gpu-test-env.sh; for c in 0 1; do echo 150000000 > $RAPL/constraint_${c}_power_limit_uw; done; setmax
QUEUE="^/bin/bash $B/post25b[.]sh" DONEFLAG=$B/.post25b-done SMCLOG=$B/smc-power-post25b.log SCLK_GUARD=1 nohup $B/clamp-watchdog-v2.sh >> $B/clamp-watchdog-post25b.out 2>&1 &
nohup $B/smc-log.sh $B/smc-power-post25b.log > /dev/null 2>&1 &
set -a; . /root/llama.cpp-benchmarking/settings/gfx906.env; set +a; export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
OUT=$B/qwen38-27b-post25b.md; echo "# post25b $(date -Is): branch r2 --no-repack, batched 8/12/16 at 2K with the CUDA op fusion on and off; production 175 / 197.5 / 202.9" > $OUT
echo "| build | fusion | 8 | 12 | 16 |" >> $OUT; echo "|---|---|---:|---:|---:|" >> $OUT
NPL=8,12,16
run() { local name=$1 P=$2 FLAG=$3; shift 3; ( export "$@"; LD_LIBRARY_PATH=$P/lib timeout 900 $P/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -b 2048 -ub 2048 -c 69632 -npp 2048 -ntg 128 -npl $NPL $FLAG > $B/post25b-$name-bb.md 2>$B/post25b-$name-bb.err )
  local cells; cells=$(grep -E '^\|' $B/post25b-$name-bb.md | grep -v 'PP \|---' | awk -F'|' '{gsub(/ /,"",$9); printf "%s | ", $9}'); echo "| $name | $cells" >> $OUT; q "post25b $name: $cells"; }
run r2-nr1-fusion-on  /opt/llama.cpp-gfx906-master-r2 --no-repack DUMMY=1
run r2-nr1-fusion-off /opt/llama.cpp-gfx906-master-r2 --no-repack GGML_CUDA_DISABLE_FUSION=1
run r2-nr1-no-custom-ar /opt/llama.cpp-gfx906-master-r2 --no-repack GGML_ENABLE_CUSTOM_AR=0
run prod-fusion-on  /opt/llama.cpp-prod "" DUMMY=1
run prod-fusion-off /opt/llama.cpp-prod "" GGML_CUDA_DISABLE_FUSION=1
NPL=16,12,8
{ echo; echo "## reverse order -npl 16,12,8: is the 8-slot loss the after-load warm-up on the first cell?"; echo "| build | fusion | 16 | 12 | 8 |"; echo "|---|---|---:|---:|---:|"; } >> $OUT
run r2-nr1-rev /opt/llama.cpp-gfx906-master-r2 --no-repack DUMMY=1
run prod-rev /opt/llama.cpp-prod "" DUMMY=1
run r2-nr0-rev /opt/llama.cpp-gfx906-master-r2 "" DUMMY=1
echo "# done $(date -Is)" >> $OUT
restore; pkill -f "smc-log[.]sh"; touch $B/.post25b-done; q "=== post25b done"
