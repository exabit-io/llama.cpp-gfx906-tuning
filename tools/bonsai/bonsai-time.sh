#!/bin/bash
# bonsai-time.sh [TAG] -- timing of Ternary-Bonsai-2.5-27B-Q2_0 on the installed package (mx-llama-cpp-gfx906 1.1.0)
# at the service's cells (REQUIREMENTS: 64K is the floor; design points 4 x 64K and 1 x 255K; agreed operating point
# 8 slots, read at 8 x 64K).
#
# bonsai-check.sh (2026-11-28 03:31-03:48 UTC) decided the arms: one die and four dies with layer split compute the
# model (same greedy text, perplexity 9.3516 in both); FOUR DIES WITH TENSOR SPLIT ABORT (ggml-backend-meta.cpp:1610,
# a reshape of a lane-split activation into the model's 1024-wide Hadamard blocks), so the way the service runs the
# Q8_0 model is not available for this model on this tree and is not timed. (This header first named b2t as the main
# arm; it was rewritten after the check and before the first timing run.)
#
# Arms (all the package's llama-batched-bench, -fa on, f16 KV, -b 2048 -ub 2048, 125 W per die, the environment of
# every survey cell):
#   b2s   Bonsai Q2_0, ONE die (--device rocm0): the 7.7 GB file leaves room for context on a single 32 GB die
#   b2l   Bonsai Q2_0, four dies, -sm layer: the layers and their KV spread over the dies, one die computing at a time
#   q8t   Qwen3.8-27B Q8_0, four dies, -sm tensor (only if named in CELLS)
# Cells (slots:depth:ntg:arms), N=3 runs per arm, arms of a cell in a fresh random order per block. Stage 1 (default):
#   1:65536:1024:b2s,b2l     one die against four (layer split) on one stream at the floor depth
#   4:65536:1024:b2l,b2s     design point (multi-user); b2s is a FIT PROBE: if the die cannot hold it the run fails fast
# Stage 2 (CELLS in the environment): 2:65536:1024:b2s 3:65536:1024:b2s 8:65536:512:b2l 1:260864:1024:b2l,b2s
# The Q8_0 reference is NOT run again: the same package was timed at 4 x 65536, 1 x 260864 and 8 x 65536 on
# 2026-11-27/28 in the acceptance session (/root/rebase-v110/screen-acc-*.tsv arm pkg, and the clean partial rerun
# screen-dp2-s4d65536.tsv); the report script prints those values beside the new ones. That is a comparison across
# sessions of one boot: decode at full clock repeats within 0.5 % between sessions here, so it carries differences of
# several percent and more, and nothing smaller.
#
# READING, WRITTEN BEFORE THE FIRST RUN. This is a benchmark, not an acceptance: the result is the table.
#   (V)  A cell/arm is read only if its runs agree: prefill spread (max - min over the mean) at most 0.5 %, decode
#        spread at most 1 %. Otherwise it is reported as not readable and the cell is run once more.
#   (T1) Per cell and arm: decode tok/s per slot and aggregate, prefill tok/s, peak VRAM per die, mean DC power in the
#        decode phase and decode tokens per joule (approximate: whole-machine DC from the SMC, 5 s samples).
#        Decode per slot is read against R3.1's 12 tok/s per request and said either way.
#   (T2) Against Q8_0 with tensor split (other session): the ratio is given; "faster" or "slower" is said only
#        beyond 3 %.
#   (T3) One die against four with layer split at 1 x 65536 (paired by block, sub-stats.py, 2 % line): reported with
#        its interval. Layer split cannot be faster than one die by construction (one die computes at a time); what
#        it buys is context. If it is slower than one die by a MAJOR PENALTY, the shape for this model is one-die
#        instances wherever the context fits one die, and four of them at once is the next measurement.
#   (T4) A fit probe that fails is reported as "does not fit on one die" with the allocator's line, not as an error.
#   Nothing here says anything about answer quality; that is bonsai-check.sh's perplexity and the model card.
set -u
W=/root/bonsai-bench; R=/root/rocm-tests/bench; TAG=${1:-t1}; N=${N:-3}
TL=/root/llama.cpp-benchmarking/tools; CM=$TL/cell-metrics.py; NF=$TL/assert-no-fa-fallback.sh
PKG=/opt/mx-llama.cpp; MB=/root/models/Ternary-Bonsai-2.5-27B-Q2_0.gguf; MQ=/root/models/Qwen3.8-27B-Q8_0.gguf
D4="--device rocm0,rocm1,rocm2,rocm3"
CELLS=${CELLS:-"1:65536:1024:b2s,b2l 4:65536:1024:b2l,b2s"}
P=$W/time-$TAG.progress; DONE=$W/.time-$TAG-done; rm -f $DONE
log(){ echo "$(date -Is) [time] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
model(){ case $1 in q8t) echo $MQ;; *) echo $MB;; esac; }
devs(){ case $1 in b2s) echo "--device rocm0";; b2l) echo "$D4 -sm layer";; *) echo "$D4 -sm tensor";; esac; }
if [ "${DRY:-0}" = 1 ]; then echo "cells: $CELLS; N=$N"; exit 0; fi
for p in llama-server llama-batched-b llama-perplexit llama-completi test-backend-op llama-cli cmake ninja; do
  pgrep -x $p >/dev/null && { log "FATAL: $p is running"; exit 1; }
done
ls $W/screen-$TAG-*.tsv >/dev/null 2>&1 && { log "FATAL: result files of tag $TAG exist"; exit 1; }
[ -f $MB ] || { log "FATAL: $MB missing"; exit 1; }
log "package $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906), source $(cut -c1-9 /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT); cells: $CELLS; N=$N; load $(cut -d' ' -f1-3 /proc/loadavg)"
. $R/gpu-test-env.sh
export GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1
BPID=""; EVS=""; VRS=""
cleanup(){ trap - INT TERM EXIT; kpid "${BPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; kpid "${VRS:-}"; touch $W/ev-$TAG.log.stop; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/time-$TAG-clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/time-$TAG-smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $W/time-$TAG-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/time-$TAG-smc.log >/dev/null 2>&1 &
SMCL=$!
bash /root/rebase-v110/evsample.sh $W/ev-$TAG.log $W/ev-$TAG.log.stop & EVS=$!
( while :; do echo "$(date +%s) $(cat /sys/class/drm/card[0-9]/device/mem_info_vram_used | tr '\n' ' ')"; sleep 5; done ) >> $W/time-$TAG-vram.log 2>/dev/null &
VRS=$!
sleep 6
log "$(state_line)"
for cell in $CELLS; do
  S=$(echo $cell | cut -d: -f1); D=$(echo $cell | cut -d: -f2); NTG=$(echo $cell | cut -d: -f3); arms=$(echo $cell | cut -d: -f4 | tr ',' ' ')
  ct=s${S}d$D; O=$W/screen-$TAG-$ct; mkdir -p $O; TSV=$W/screen-$TAG-$ct.tsv; RUNS=$W/runs-$TAG.tsv
  dead=""
  for blk in $(seq 1 $N); do
    order=$(echo $arms | tr ' ' '\n' | shuf | tr '\n' ' '); log "$S x $D block $blk order: $order"
    for arm in $order; do
      case " $dead " in *" $arm "*) continue;; esac
      out=$O/$arm-$blk.md; t0=$(date +%s)
      env -u LD_LIBRARY_PATH AMD_COMGR_CACHE_DIR=/root/.cache/comgr-bonsai-$arm timeout 7200 \
        $PKG/bin/llama-batched-bench -m $(model $arm) $(devs $arm) -ngl all \
        -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $(( S*(D+1280) )) -npp $D -ntg $NTG -npl $S \
        > $out 2>${out%.md}.log < /dev/null &
      BPID=$!
      wait $BPID; rc=$?
      BPID=""; t1=$(date +%s)
      $NF "${out%.md}.log" >/dev/null 2>&1 || { log "FA FALLBACK in $arm b$blk"; continue; }
      case $(devs $arm) in *"-sm tensor"*) grep -qiE 'NCCL init failed|probe AllReduce failed' "${out%.md}.log" && { log "RCCL NOT UP in $arm b$blk"; continue; };; esac
      if ! m=$(python3 -I $CM "$out" $D $S 2>>$P); then
        log "$S x $D $arm b$blk: NO RESULT (rc=$rc, $(( t1 - t0 )) s): $(grep -iE 'out of memory|failed to allocate|unable to allocate|alloc|error' "${out%.md}.log" | tail -2 | tr '\n' ' ' | cut -c1-300)"
        dead="$dead $arm"; continue
      fi
      printf "%s\t%s\t%s\t%s\n" "$S x $D" "$arm" "$blk" "$m" >> $TSV
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$S" "$D" "$NTG" "$arm" "$blk" "$t0" "$t1" "$m" >> $RUNS
      log "$S x $D $arm b$blk: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}') ($(( t1 - t0 )) s)"
    done
  done
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${SMCL:-}"; kpid "${VRS:-}"; touch $W/ev-$TAG.log.stop; wait $EVS 2>/dev/null; touch $DONE; sleep 6; kpid "${WDOG:-}"; restore
python3 -I $W/bonsai-report.py $W $TAG 2>&1 | tee -a $P
log "--- four dies with layer split against one die, paired by block (sub-stats.py, 2 % line) ---"
SUBW=$W PAIRS="b2l:b2s" LINE=2 python3 -I /root/rebase-v110/sub-stats.py $TAG 2>&1 | sed -n '/ against /,$p' | tee -a $P
log "=== BONSAI-TIME $TAG DONE ==="
