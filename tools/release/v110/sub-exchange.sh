#!/bin/bash
# sub-exchange.sh -- positive check for rule (b) of substrate-v110.sh: which library carries the exchange between the
# dies in each arm, read from a verbose (-lv 4) run that is NOT timed (the timed runs log at the default level, where
# only failures show). Same environment as screen.sh. Correctness only.
# READING, before the run: every arm must show the RCCL/NCCL initialisation and no failure line; an arm without it
# runs its large tensors on another path and its tensor-split rows could not be read as a property of its code.
set -u
W=/root/rebase-v110; RT=/opt/rocm/core-10.0; M=/root/models/Qwen3.8-27B-Q8_0.gguf; P=$W/sub-exchange.progress; O=$W/sub-exchange; mkdir -p $O
log(){ echo "$(date -Is) [sub-exchange] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is running"; exit 1; }
. /root/rocm-tests/bench/gpu-test-env.sh
export GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
trap 'restore' EXIT
setmax
for spec in mxxm=/root/build-sub-mxxm merge=/root/build-sub-merge master=/root/build-master-v110; do
  a=${spec%%=*}; bd=${spec#*=}
  LD_LIBRARY_PATH=$bd/bin:$RT/lib timeout 900 $bd/bin/llama-batched-bench -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on -ctk f16 -ctv f16 \
    -b 2048 -ub 2048 -c 8192 -npp 512 -ntg 8 -npl 4 -lv 4 > $O/$a.md 2> $O/$a.log
  log "$a rc=$?: $(grep -a -i -E 'nccl|rccl|allreduce' $O/$a.log | sed -E 's/^[0-9. ]*[A-Z] //' | sort | uniq -c | sort -rn | head -6 | tr -s ' ' | tr '\n' ';' | cut -c1-520)"
done
log "=== SUB-EXCHANGE DONE ==="
