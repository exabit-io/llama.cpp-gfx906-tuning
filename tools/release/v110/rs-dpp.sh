#!/bin/bash
# rs-dpp.sh -- step 3 after rs-bisect.sh. The first commit at which "replay" and "snapshots on the recurrent kernel"
# write different text is term 23 (76a525c37, DPP-based warp reductions on GCN: an xor butterfly taken in the order
# 1, 2, 4, 8, 16, 32 where the generic one runs 32 ... 1, so sums round differently). Both arms run the recurrent GDN
# kernel; its replay loop and its live loop carry the same source arithmetic and upstream's comment says they are meant
# to match bit for bit. Correctness only (dies at 1000 MHz): no rate is a number.
# Experiment: master-v110 with ONE line added, `#define GGML_GCN_NO_DPP` at the top of gated_delta_net.cu (the opt-out
# term 28 uses for the flash-attention kernels), the two arms, the three folds off as in rs-bisect.sh.
# READING, before the run:
#   same text on all four requests = the two loops of the GDN kernel stop matching bit for bit when its reductions are
#     the DPP ones. Then the GDN kernel is where to look (what the compiler does differently in the two loops), and the
#     opt-out is a candidate fix whose decode cost has to be timed at full clock before it is taken.
#   still different = the DPP reductions act through ANOTHER kernel that the two graphs use differently; opt further
#     files out one at a time to name it.
set -u
W=/root/rebase-v110; G=/root/exabit-llama.cpp; RT=/opt/rocm/core-10.0; WT=/root/wt-bisect; B=/root/build-bisect; P=$W/rs-dpp.progress; O=$W/rs-dpp
log(){ echo "$(date -Is) [rs-dpp] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is running"; exit 1; }
mkdir -p $O
git -C $WT checkout -q --detach master-v110 || { log "FATAL: checkout"; exit 1; }
f=$WT/ggml/src/ggml-cuda/gated_delta_net.cu
grep -q '^#define GGML_GCN_NO_DPP' $f || sed -i '1i #define GGML_GCN_NO_DPP' $f
log "tree: master-v110 $(git -C $WT rev-parse --short HEAD) + $(git -C $WT diff --shortstat | tr -s ' ')"
if ! UIVER=b12774 bash $W/build-rocm.sh $WT $B $O/build.log $RT > $O/build.out 2>&1; then log "FATAL: build failed: $(tail -1 $O/build.out)"; exit 1; fi
GGML_CUDA_NORM_Q8=0 GGML_CUDA_ADD_NORM=0 GGML_CUDA_GDN_PREFUSE=0 RT=$RT ARMS="replay snaprec" bash $W/rs-modes3.sh $B $O/foldsoff > $O/foldsoff.out 2>&1
log "folds off: $(grep -E 'text (long|[0-9])' $O/foldsoff/progress | sed -E 's/.*text ([a-z0-9]+): .*replay ([0-9a-f]+|none) snaprec ([0-9a-f]+|none).*/\1 \2|\3/' | tr '\n' ' ')"
RT=$RT ARMS="replay snaprec" bash $W/rs-modes3.sh $B $O/foldson > $O/foldson.out 2>&1
log "folds on (default): $(grep -E 'text (long|[0-9])' $O/foldson/progress | sed -E 's/.*text ([a-z0-9]+): .*replay ([0-9a-f]+|none) snaprec ([0-9a-f]+|none).*/\1 \2|\3/' | tr '\n' ' ')"
git -C $WT checkout -q -- ggml/src/ggml-cuda/gated_delta_net.cu
log "=== RS-DPP DONE ==="
