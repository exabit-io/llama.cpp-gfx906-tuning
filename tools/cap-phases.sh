#!/bin/bash
# TODO 13: phase-separated cap curve on the production build: per cap, decode-only (batched-bench tp4 16 slots at 2K) and prefill-only
# (llama-bench pp2048 tp4). Caps 200/170/140/125/85 then 200 again. Watchdog must run with SCLK_GUARD=0 for this job.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-cap-phases; P=/opt/llama.cpp-prod; DIES="0b 0e 1b 1e"
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
capfile() { ls /sys/bus/pci/devices/0000:$1:00.0/hwmon/hwmon*/power1_cap; }
setcap() { local w=$1 d; for d in $DIES; do echo $((w*1000000)) > $(capfile $d); done; sleep 2; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; setcap 200; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_TP_AR_MAX_NE=20481
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md
{ echo "# $TAG  $(date -Is)  production build tp4: decode-only (16 slots, 2K) and prefill-only (pp2048) per cap"; echo "| cap W | decode 16 slots tok/s | prefill pp2048 tok/s | mean die W during decode | during prefill |"; echo "|---:|---:|---:|---:|---:|"; } > $OUT
for CAP in 200 170 140 125 85 200; do setcap $CAP; log "cap $CAP"
  t0=$(date +%s); $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 34816 -npp 2048 -ntg 128 -npl 16 > $B/$TAG-$CAP-bb.md 2>/dev/null; t1=$(date +%s)
  dec=$(grep -E '^\| *2048 ' $B/$TAG-$CAP-bb.md | awk -F'|' '$4+0==16 {gsub(/ /,"",$9); print $9}')
  $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 2048 -ub 2048 -b 2048 -n 0 -r 3 -o md > $B/$TAG-$CAP-lb.md 2>/dev/null; t2=$(date +%s)
  pre=$(grep -E '^\| qwen.*pp2048' $B/$TAG-$CAP-lb.md | awk -F'|' '{gsub(/^ +| +$/,"",$12); print $12}')
  wd=$(awk -v a=$t0 -v b=$t1 'BEGIN{s=0;n=0} {cmd="date -d "$1" +%s"; } END{}' /dev/null); 
  echo "| $CAP | ${dec:-fail} | ${pre:-fail} | see clocks.txt $(date -d @$t0 +%T)-$(date -d @$t1 +%T) | $(date -d @$t1 +%T)-$(date -d @$t2 +%T) |" >> $OUT; log "cap $CAP: decode $dec prefill $pre"
done
setcap 200
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
