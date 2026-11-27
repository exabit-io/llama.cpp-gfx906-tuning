#!/bin/bash
# master-checks.sh -- two controls for the clamped correctness run of master-v110 (queue-clamped.sh, 2026-11-26).
# Correctness only (dies at 1000 MHz): no rate printed here is a number.
# (A) test-backend-ops: ROCm3 failed ONE case of 21871, SOLVE_TRI(f32, lhs [128,128,4,1], rhs [32,128,4,1]) with error
#     1.66e-7 against the test's 1.0e-7; first failure of that case in 42 logs of two cycles; master does not touch the
#     solver. The test draws fresh random inputs in every run. Control: the op alone, RUNS times per die, on master and
#     on the merge build.
#     READING (before the run): failures on more than one die, or on the merge build too = the test's tolerance against
#     random 128 x 128 systems, not our code and not die 3. Failures only on ROCm3 in both builds = die 3 is suspect
#     (hardware or the latched clock): stop and look. Failures only in master = ours: stop and look.
# (B) rs-modes3 on master: replay and snapshots-on-the-recurrent-kernel wrote DIFFERENT text (rule 3), where the merge
#     writes the same. Master folds the GDN producer into snapshot ops and not into replay ops, and that fold is not
#     bit-exact. Control: the two arms again with every fold off (GGML_CUDA_NORM_Q8=0 GGML_CUDA_ADD_NORM=0
#     GGML_CUDA_GDN_PREFUSE=0, as fold-check.sh's "folds-off").
#     READING (before the run): the same text on all four requests = the difference in the default run is the folds'
#     rounding, the rollback is not involved. Still different = a real difference in master's rollback path: nothing
#     ships before it is found.
set -u
W=/root/rebase-v110; RT=/opt/rocm/core-10.0; RUNS=${RUNS:-12}; P=$W/master-checks.progress
log(){ echo "$(date -Is) [master-checks] $*" | tee -a $P; }
pgrep -x llama-server >/dev/null && { log "FATAL: a llama-server is running"; exit 1; }
pgrep -x cmake >/dev/null && { log "FATAL: a build is running"; exit 1; }
mkdir -p $W/master-checks
for b in master merge; do
  B=/root/build-$b-v110
  for d in 0 1 2 3; do
    fail=0; errs=""
    for r in $(seq 1 $RUNS); do
      out=$W/master-checks/solve-$b-$d-$r.log
      LD_LIBRARY_PATH=$B/bin:$RT/lib timeout 600 $B/bin/test-backend-ops -b ROCm$d -o SOLVE_TRI > $out 2>&1 || true
      n=$(sed 's/\x1b\[[0-9;]*m//g' $out | grep -c 'SOLVE_TRI.*FAIL')
      [ "$n" -gt 0 ] && { fail=$((fail + 1)); errs="$errs $(sed 's/\x1b\[[0-9;]*m//g' $out | grep -o 'ERR = [0-9.]*' | head -1 | cut -d' ' -f3)"; }
      grep -q 'tests passed' $out || { log "  $b ROCm$d run $r: no summary line"; }
    done
    log "(A) $b ROCm$d: $fail of $RUNS runs with a failed SOLVE_TRI case; errors:${errs:- none}; cases per run: $(sed 's/\x1b\[[0-9;]*m//g' $W/master-checks/solve-$b-$d-1.log | grep -o '[0-9]*/[0-9]* tests passed' | head -1)"
  done
done
log "(B) rs-modes3, master, folds off, arms replay and snaprec"
GGML_CUDA_NORM_Q8=0 GGML_CUDA_ADD_NORM=0 GGML_CUDA_GDN_PREFUSE=0 RT=$RT ARMS="replay snaprec" bash $W/rs-modes3.sh /root/build-master-v110 $W/master-checks/rs-foldsoff > $W/master-checks/rs-foldsoff.out 2>&1
log "(B) rc=$?"
grep -E 'text (long|[0-9])' $W/master-checks/rs-foldsoff/progress | cut -c40-200 | while read -r l; do log "(B) $l"; done
log "=== MASTER-CHECKS DONE ==="
