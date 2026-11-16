#!/bin/bash
# deb100-chain.sh -- the ROCm 10.0-stream package of 2026-11-16 (last-deb.txt): package test (deb-test-llama-100.sh,
# which installs it), then, only if the test is clean, the design-point acceptance on the installed package
# (rocm-ab4.sh, about 3 hours). The box is left on the 10.0-stream package: publish and dogfood-llama-100.sh follow,
# and the dogfood puts the 10.1 package back.
set -u
W=/root/rocm101-migration-20261114; O=$W/deb-test-llama-100r1
pgrep -x llama-batched-b >/dev/null && { echo "a timing run is in progress" >&2; exit 1; }
case $(cat /root/mx-llama-deb/last-deb.txt) in *.rocm10.0.gfx906.*) ;; *) echo "last-deb.txt is not the 10.0-stream package" >&2; exit 1;; esac
bash $W/deb-test-llama-100.sh $O > $O.out 2>&1
P=$O/progress
ok=1
grep -q 'apt-get install rc=0 -> .*rocm10\.0\.gfx906' $P || ok=0
grep -q 'without DT_RPATH=/opt/mx-llama.cpp/lib:/opt/rocm/core-10.0/lib: 0$' $P || ok=0
grep -q 'ggml/llama libraries outside /opt/mx-llama.cpp/lib: 0; ROCm libraries outside /opt/rocm/core-10.0: 0 of' $P || ok=0
grep -q 'pr-test deb rc=0' $P || ok=0
grep -q 'deb MTP np1 rc=0' $P || ok=0
grep -q 'deb MTP np4 rc=0' $P || ok=0
if [ $ok != 1 ]; then echo "$(date -Is) [chain] package test NOT clean: acceptance not started" | tee -a $P; exit 1; fi
echo "$(date -Is) [chain] package test clean: starting rocm-ab4.sh" | tee -a $P
bash $W/rocm-ab4.sh > $W/rocm-ab4.out 2>&1
echo "$(date -Is) [chain] rocm-ab4 rc=$?" | tee -a $P
