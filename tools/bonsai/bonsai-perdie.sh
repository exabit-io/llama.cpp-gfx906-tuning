#!/bin/bash
# bonsai-perdie.sh [TAG] -- Ternary-Bonsai-2.5-27B-Q2_0 as SEPARATE INSTANCES, ONE PER DIE (lead, 2026-11-28: "The
# model is small enough where you don't need to use layer split or tensor split, just run a separate instance on each
# die"). Installed package mx-llama-cpp-gfx906 1.1.0, llama-batched-bench, -fa on, f16 KV, -b 2048 -ub 2048, -t 8,
# 125 W per die, 64K per slot (REQUIREMENTS R2.2: the floor), 1024 generated tokens per slot.
#
# Steps, in this order:
#   probe     what one die holds: for S = 1..5 slots of 64K, and for one slot of 262144, a tiny run with the full
#             context allocated (-npp 256 -ntg 8): loads or not, VRAM in use. Functional, no rate is quoted.
#   quad:S    FOUR instances at once, one per die (rocm0..rocm3), each with S slots x 65536. Total slots 4 x S.
#   solo:S    ONE instance on die 0, the other three dies idle (what the neighbours cost).
# Default plan: probe, quad:1, quad:2, solo:1, quad:3 with N=2 runs per step (a quad run gives four per-die samples;
# two runs take 13 to 35 minutes per step at one die's prefill rate, so N=3 would not finish tonight).
# A quad:S whose S does not fit by the probe is skipped.
#
# The Q8_0 reference is NOT run again: Qwen3.8-27B Q8_0 on four dies with tensor split, same package, was timed at
# 4 x 65536 and 8 x 65536 on 2026-11-27/28 (acceptance session; /root/rebase-v110/screen-acc-*.tsv arm pkg and the
# partial clean rerun screen-dp2-s4d65536.tsv). Other session of the same boot: carries differences of several
# percent and more, nothing smaller.
#
# READING, WRITTEN BEFORE THE FIRST RUN. A benchmark: the result is the table.
#   (V)  A step is read only if, for every die, its two runs agree within 0.5 % in prefill and 1 % in decode;
#        otherwise the step is reported as not readable and run once more.
#   (R1) Per step: per die decode tok/s per slot and prefill tok/s; aggregate decode (sum over dies of S x per-slot)
#        and aggregate prefill (sum over dies); peak VRAM per die; mean DC power in the decode phase and decode
#        tokens per joule (whole-machine DC from the SMC, 5 s samples: approximate).
#   (R2) Per-slot decode against R3.1's 12 tok/s per request, said either way, per step.
#   (R3) Against Q8_0 with tensor split at the same total (quad:1 = 4 x 64K, quad:2 = 8 x 64K): per-slot decode and
#        aggregate prefill as ratios; "faster"/"slower" said only beyond 3 %.
#   (R4) Neighbours: die 0 in quad:1 against solo:1, ratio of the means; said as a cost only beyond 2 %.
#   (R5) The probe gives the capacity table (slots of 64K per die; whether 262144 fits). A failed allocation is a
#        result ("does not fit"), not an error.
#   Quality is not judged here: perplexity 9.35 (Qwen3.8-27B Q8_0: 5.61) is bonsai-check.sh's reading.
set -u
W=/root/bonsai-bench; R=/root/rocm-tests/bench; TAG=${1:-pd1}; N=${N:-2}; D=65536; NTG=1024
TL=/root/llama.cpp-benchmarking/tools; CM=$TL/cell-metrics.py; NF=$TL/assert-no-fa-fallback.sh
PKG=/opt/mx-llama.cpp; MB=/root/models/Ternary-Bonsai-2.5-27B-Q2_0.gguf
STEPS=${STEPS:-"probe quad:1 quad:2 solo:1 quad:3"}
P=$W/perdie-$TAG.progress; DONE=$W/.perdie-$TAG-done; O=$W/perdie-$TAG; RUNS=$W/perdie-$TAG.tsv
log(){ echo "$(date -Is) [perdie] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
if [ "${DRY:-0}" = 1 ]; then echo "steps: $STEPS; N=$N"; exit 0; fi
for p in llama-server llama-batched-b llama-perplexit llama-completi test-backend-op llama-cli cmake ninja; do
  pgrep -x $p >/dev/null && { log "FATAL: $p is running"; exit 1; }
done
[ -e $RUNS ] && { log "FATAL: result file of tag $TAG exists"; exit 1; }
[ -f $MB ] || { log "FATAL: $MB missing"; exit 1; }
mkdir -p $O; rm -f $DONE
log "package $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906), source $(cut -c1-9 /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT); steps: $STEPS; N=$N; load $(cut -d' ' -f1-3 /proc/loadavg)"
. $R/gpu-test-env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
B0=""; B1=""; B2=""; B3=""; EVS=""; VRS=""
# kben: a bench runs as subshell -> timeout -> llama-batched-bench; killing only the subshell leaves the benchmark on
# the die (seen 2026-11-28 04:17 UTC: four orphans after a TERM, killed by pid). Signal the subshell's child first.
kben(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && { pkill -TERM -P "$1" 2>/dev/null; kill "$1" 2>/dev/null; }; return 0; }
cleanup(){ trap - INT TERM EXIT; kben "$B0"; kben "$B1"; kben "$B2"; kben "$B3"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; kpid "${VRS:-}"; touch $W/ev-$TAG.log.stop; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/perdie-$TAG-clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/perdie-$TAG-smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $W/perdie-$TAG-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/perdie-$TAG-smc.log >/dev/null 2>&1 &
SMCL=$!
bash /root/rebase-v110/evsample.sh $W/ev-$TAG.log $W/ev-$TAG.log.stop & EVS=$!
( while :; do echo "$(date +%s) $(cat /sys/class/drm/card[0-9]/device/mem_info_vram_used | tr '\n' ' ')"; sleep 2; done ) >> $W/perdie-$TAG-vram.log 2>/dev/null &
VRS=$!
sleep 6
log "$(state_line)"
bench(){ # bench DIE S NPP NTG CTX OUT  (background; prints nothing)
  env -u LD_LIBRARY_PATH AMD_COMGR_CACHE_DIR=/root/.cache/comgr-bonsai-die$1 timeout 7200 \
    $PKG/bin/llama-batched-bench -m $MB --device rocm$1 -ngl all -t 8 \
    -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $5 -npp $3 -ntg $4 -npl $2 > $6 2>${6%.md}.log < /dev/null
}
FIT=""
for step in $STEPS; do
  mode=${step%%:*}; S=${step#*:}
  if [ "$mode" = probe ]; then
    for spec in 1:66816 2:133632 3:200448 4:267264 5:334080 1:262144; do
      s=${spec%%:*}; ctx=${spec#*:}; out=$O/probe-$s-$ctx.md; v0=$(date +%s)
      bench 0 $s 256 8 $ctx $out & B0=$!
      wait $B0; rc=$?; B0=""; v1=$(date +%s)
      peak=$(awk -v a=$v0 -v b=$v1 '$1 >= a && $1 <= b { n++; if ($2 > m) m = $2 } END { if (n) printf "%.1f", m / 1073741824; else printf "n/a" }' $W/perdie-$TAG-vram.log)
      if [ $rc = 0 ] && grep -qE '^\|\s*256\s' $out; then
        log "probe: $s slot(s), context $ctx on one die: LOADS and runs, peak VRAM $peak GiB"; [ $ctx != 262144 ] && FIT="$FIT $s"
      else
        log "probe: $s slot(s), context $ctx on one die: DOES NOT FIT or fails (rc=$rc, peak VRAM $peak GiB): $(grep -iE 'out of memory|failed to allocate|unable to allocate|failed to create|alloc' ${out%.md}.log | tail -1 | cut -c1-220)"
      fi
    done
    log "probe: slots of 64K that fit one die:${FIT:- none}"
    continue
  fi
  if [ -n "$FIT" ]; then case " $FIT " in *" $S "*) ;; *) log "$step: skipped, $S slots of 64K do not fit one die by the probe"; continue;; esac; fi
  ctx=$(( S * (D + 1280) ))
  for blk in $(seq 1 $N); do
    t0=$(date +%s)
    if [ "$mode" = quad ]; then
      bench 0 $S $D $NTG $ctx $O/$mode-$S-die0-$blk.md & B0=$!
      bench 1 $S $D $NTG $ctx $O/$mode-$S-die1-$blk.md & B1=$!
      bench 2 $S $D $NTG $ctx $O/$mode-$S-die2-$blk.md & B2=$!
      bench 3 $S $D $NTG $ctx $O/$mode-$S-die3-$blk.md & B3=$!
      wait $B0; wait $B1; wait $B2; wait $B3; B0=""; B1=""; B2=""; B3=""; dies="0 1 2 3"
    else
      bench 0 $S $D $NTG $ctx $O/$mode-$S-die0-$blk.md & B0=$!
      wait $B0; B0=""; dies="0"
    fi
    t1=$(date +%s); line=""
    for d in $dies; do
      out=$O/$mode-$S-die$d-$blk.md
      $NF "${out%.md}.log" >/dev/null 2>&1 || { log "FA FALLBACK in $step die $d run $blk"; continue; }
      if ! m=$(python3 -I $CM "$out" $D $S 2>>$P); then log "$step die $d run $blk: NO RESULT: $(tail -2 ${out%.md}.log | tr '\n' ' ' | cut -c1-240)"; continue; fi
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$mode" "$S" "$D" "$NTG" "$d" "$blk" "$t0" "$t1" "$m" >> $RUNS
      line="$line die$d $(echo "$m" | awk -F'\t' '{printf "%.3f/%.1f",$1,$2}')"
    done
    log "$step run $blk ($(( t1 - t0 )) s), decode per slot / prefill:$line"
  done
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${SMCL:-}"; kpid "${VRS:-}"; touch $W/ev-$TAG.log.stop; wait $EVS 2>/dev/null; touch $DONE; sleep 6; kpid "${WDOG:-}"; restore
python3 -I $W/perdie-report.py $W $TAG 2>&1 | tee -a $P
log "=== BONSAI-PERDIE $TAG DONE ==="
