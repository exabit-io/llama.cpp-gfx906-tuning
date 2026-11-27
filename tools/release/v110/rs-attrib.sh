#!/bin/bash
# rs-attrib.sh -- where the text difference between "replay" and "snapshots on the recurrent kernel" on master comes from.
# Correctness only (dies at 1000 MHz): no rate printed here is a number.
# Known (2026-11-26): the merge build (RCCL off) writes the same text in both arms on all four requests. master-v110
# (RCCL on by its first commit, 22 Exabit commits) writes different text, and still does with its three folds off
# (master-checks.sh (B): "find it before anything ships"). test-llama-context's rollback cases pass on master in both
# modes, so the rollback machinery itself is not what is in doubt; what is not explained is which operation computes
# differently in the two graphs.
# Master differs from that merge build in two things: the 22 commits, and the exchange between the dies (RCCL above
# GGML_TP_AR_MAX_NE elements, the custom allreduce below; the merge build without RCCL has one path for every size).
# Step 1 (this script): the merge CODE built with RCCL on (/root/build-sub-merge), the two arms,
#   (a) with its own default environment, (b) with master's exchange settings in the environment
#   (GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481).
# READING, before the run:
#   (a) or (b) different text  = the exchange path, not an Exabit commit: the two graphs hand the exchange tensors of
#       different sizes, and the two exchange implementations add the four dies' parts in a different order. Then the
#       sizes are read from the -lv 4 logs to name the tensor.
#   (a) and (b) the same text  = one of the 22 commits: bisect them with these two arms (step 2, builds needed).
set -u
W=/root/rebase-v110; RT=/opt/rocm/core-10.0; B=/root/build-sub-merge; P=$W/rs-attrib.progress; O=$W/rs-attrib
log(){ echo "$(date -Is) [rs-attrib] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is running"; exit 1; }
[ -x $B/bin/llama-server ] || { log "FATAL: $B is not built"; exit 1; }
mkdir -p $O
run(){ # run NAME [VAR=VALUE ...]
  local name=$1; shift
  env "$@" RT=$RT ARMS="replay snaprec" bash $W/rs-modes3.sh $B $O/$name > $O/$name.out 2>&1
  log "$name rc=$?: $(grep -E 'text (long|[0-9])' $O/$name/progress | sed -E 's/.*text ([a-z0-9]+): .*replay ([0-9a-f]+) snaprec ([0-9a-f]+).*/\1 \2|\3/' | tr '\n' ' ')"
  log "$name exchange lines (replay arm): $(grep -h -i -E 'allreduce|nccl|rccl' $O/$name/replay-server.log 2>/dev/null | sed -E 's/^[0-9. ]*[A-Z] //' | sort | uniq -c | sort -rn | head -4 | tr '\n' ';' | cut -c1-300)"
}
run a LLAMA_RS_DUMMY=0
run b GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481
log "=== RS-ATTRIB DONE ==="
