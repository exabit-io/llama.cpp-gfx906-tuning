#!/bin/bash
# queue-rel-v110.sh WANT -- everything that needs the GPUs between "the final master is built and packaged" and
# "publish", unattended and in the order of the -r2 release (tools/rocm101/README.md), one job after the other:
#   final    FOLD=1 queue-final.sh on the ROCm 10.0 source build of WANT (/root/build-final-v110): test-backend-ops on
#            four dies, PPL 16K x 6, the fold check, the fixture tests in both rollback modes, MTP row paths, the
#            carried MTP row, the rollback modes
#   ds41     DeepSeek-V4.1 on the same build: test-llama-archs -a deepseek41, dsv41-test.sh full (upstream-layout
#            file: -sm layer text, PPL 2K x 4, -sm tensor text, server), ds41-dspark.sh
#   debtest  deb-test-v110.sh: INSTALLS the deb named in /root/mx-llama-deb/last-deb.txt, ELF properties, the suite,
#            MTP row paths, DeepSeek-V4.1 on the package
#   tbo      tbo-pkg-v110.sh: test-backend-ops with the installed package's GPU library (test binary of the ROCm 10.1
#            source build of WANT, /root/build-final-v110-101)
#   accept   accept-v110.sh: the timing acceptance and the confirmation of the two defaults (about 6 h; rule in its header)
#   mtp      mtp-accept-v110.sh: MTP at depth on the installed package (about 55 min; rule in its header)
# Nothing here decides anything: every job's result is read afterwards against the rule in that job's script, and the
# package is published only if all of them pass. A job that fails does not stop the next one, with two exceptions:
# a clock latch stops the chain (every rate after it would be void), and without an installed 1.1.0 package built
# from WANT the three package jobs are skipped.
# Markers .rel-done-NAME make a second start continue where the first stopped. No build runs in here.
set -u
W=/root/rebase-v110; P=$W/queue-rel.progress; WANT=${1:?usage: $0 COMMIT}
SRC=/root/wt-master-v110; B100=/root/build-final-v110; B101=/root/build-final-v110-101
log(){ echo "$(date -Is) [rel] $*" | tee -a $P; }
clamp_check(){ # 0 when every die reads the top clock at idle under perf level high
  local d m bad=0
  for d in /sys/class/drm/card[0-9]/device; do [ -f $d/pp_dpm_sclk ] && echo high > $d/power_dpm_force_performance_level; done; sleep 5
  for d in /sys/class/drm/card[0-9]/device; do [ -f $d/pp_dpm_sclk ] || continue; m=$(grep '\*' $d/pp_dpm_sclk | awk '{print $2}' | tr -dc 0-9); [ "${m:-0}" -ge 1700 ] || bad=1; done
  for d in /sys/class/drm/card[0-9]/device; do [ -f $d/pp_dpm_sclk ] && echo auto > $d/power_dpm_force_performance_level; done
  return $bad
}
job(){ # job NAME COMMAND...
  local name=$1; shift
  [ -f $W/.rel-done-$name ] && { log "SKIP $name: done before"; return 0; }
  for p in llama-server llama-batched-b llama-perplexit test-backend-op cmake; do pgrep -x $p >/dev/null && { log "FATAL: $p is running before $name"; exit 1; }; done
  clamp_check || { log "CLAMP: a die reads below 1700 MHz at idle under perf high before $name: chain stopped (reboot needed)"; exit 2; }
  log "RUN $name"
  "$@" > $W/rel-$name.out 2>&1
  log "END $name rc=$?"
  touch $W/.rel-done-$name
}
pkg_ok(){ case "$(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906 2>/dev/null)" in 1.1.0*.rocm10.1.gfx906.*) ;; *) return 1;; esac; [ "$(cat /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT 2>/dev/null)" = "$WANT" ]; }
[ "$(git -C $SRC rev-parse HEAD)" = "$WANT" ] || { log "FATAL: $SRC is not at $WANT"; exit 1; }
for b in $B100 $B101; do [ "$(cat $b/.src-commit 2>/dev/null)" = "$WANT" ] || { log "FATAL: $b is not built from $WANT"; exit 1; }; done
DEB=$(cat /root/mx-llama-deb/last-deb.txt 2>/dev/null)
[ -f "$DEB" ] && [ "$(dpkg-deb --fsys-tarfile "$DEB" | tar -xO ./usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT 2>/dev/null)" = "$WANT" ] || { log "FATAL: the deb in last-deb.txt ($DEB) is not built from $WANT"; exit 1; }
log "start: commit ${WANT:0:9}, deb $(basename "$DEB"), installed now $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906 2>/dev/null), macpro-fand $(systemctl is-active macpro-fand), up $(uptime -p)"
systemctl is-active macpro-fand >/dev/null || { log "FATAL: macpro-fand is not active"; exit 1; }
job final   env FOLD=1 bash $W/queue-final.sh $B100 final $SRC
job ds41    bash -c "
  LD_LIBRARY_PATH=$B100/bin:/opt/rocm/core-10.0/lib HSA_FORCE_FINE_GRAIN_PCIE=1 timeout 1800 $B100/bin/test-llama-archs -a deepseek41 > $W/final-ds41-archs.out 2>&1; echo \"test-llama-archs -a deepseek41 rc=\$?\"
  RT=/opt/rocm/core-10.0 bash $W/dsv41-test.sh $B100 $W/final-dsv41 full; echo \"dsv41-test rc=\$?\"
  RT=/opt/rocm/core-10.0 bash $W/ds41-dspark.sh $B100 $W/final-dspark; echo \"ds41-dspark rc=\$?\""
job debtest bash $W/deb-test-v110.sh $W/deb-test-v110
if pkg_ok; then
  job tbo     env TB=$B101 bash $W/tbo-pkg-v110.sh $W/tbo-pkg-v110
  job accept  env WANT=$WANT bash $W/accept-v110.sh
  job mtp     env WANT=$WANT bash $W/mtp-accept-v110.sh
else
  log "the installed package is not the 1.1.0 build of ${WANT:0:9} ($(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906 2>/dev/null)): tbo, accept and mtp NOT run"
fi
log "=== QUEUE-REL DONE ==="
