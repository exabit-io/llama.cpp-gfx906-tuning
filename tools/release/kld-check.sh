#!/bin/bash
# kld-check.sh -- KL divergence of the GDN L2-norm fold: logits with every fold off (base) vs the build's default (all on),
# Qwen3.8-27B Q8_0, 2K x 4 chunks at ubatch 16, old master (82868aa3b) and rebased master (ef70ef51e).
set -u
W=/root/rebase-v070; R=/root/rocm-tests/bench; O=$W/kld; mkdir -p $O
M=/root/models/Qwen3.8-27B-Q8_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
L(){ echo "$1/bin:$1/lib:/opt/rocm/lib:/opt/rocm/core-10.0/lib"; }
log(){ echo "$(date -Is) [kld] $*" | tee -a $O/progress; }
. $R/gpu-test-env.sh; setmax
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml HSA_FORCE_FINE_GRAIN_PCIE=1
DONE=$O/.done; rm -f $DONE
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$O/smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $O/watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $O/smc.log >/dev/null 2>&1 &
SMCL=$!
trap 'kill $WDOG $SMCL 2>/dev/null; restore; rm -f $O/*.bin; touch $DONE; exit 1' INT TERM
P(){ # P BUILD NAME ENV ARGS... (ENV: space-separated VAR=value list, or "-")
  local b=$1 name=$2 e=$3; shift 3
  [ "$e" = - ] && e=""
  env $e LD_LIBRARY_PATH=$(L $b) timeout 3600 $b/bin/llama-perplexity -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all -fa on \
    -b 16 -ub 16 -c 2048 --chunks 4 -f $F "$@" > $O/$name.out 2> $O/$name.log < /dev/null
}
for arm in old=/root/build-mx-both new=/root/build-master-v070; do
  n=${arm%%=*}; b=${arm#*=}
  P $b $n-base "GGML_CUDA_NORM_Q8=0 GGML_CUDA_ADD_NORM=0 GGML_CUDA_GDN_PREFUSE=0" --kl-divergence-base $O/$n-base.bin
  log "$n base (folds off): $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $O/$n-base.log | tail -1)"
  P $b $n-on - --kl-divergence-base $O/$n-base.bin --kl-divergence
  log "$n folds on vs off: $(grep -E 'Mean +KLD|Maximum KLD|99.9% +KLD|Same top p|Mean +ln\(PPL\(Q\)/PPL\(base\)\)|RMS Δp' $O/$n-on.log | tr -s ' ' | tr '\n' ';')"
  rm -f $O/$n-base.bin
done
kill $WDOG $SMCL 2>/dev/null; touch $DONE
restore
log "=== KLD DONE ==="
