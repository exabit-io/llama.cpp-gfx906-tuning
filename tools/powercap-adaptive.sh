#!/bin/bash
# Adaptive power-cap search for serving performance per watt on the production build (see powercap-adaptive.py). One llama-server
# tp4 -np 16 stays up for the whole run; caps switch live; caps go back to 200 W at the end and in the trap. ~2-2.5 h GPU.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-mxxmfh-powercap-adaptive; PORT=8092; LP=/opt/llama.cpp-mxxm-fh
. $B/gpu-test-env.sh
DIES="0b 0e 1b 1e"; capfile() { ls /sys/bus/pci/devices/0000:$1:00.0/hwmon/hwmon*/power1_cap; }
setcap() { local w=$1 d; for d in $DIES; do echo $((w*1000000)) > $(capfile $d); done; }
QP=$B/$TAG-queue.progress; : > $QP; echo $$ > $B/$TAG.pid
log() { echo "$(date -Is) $*" >> $QP; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; pkill -f "^$LP/bin/llama-server .*--port $PORT" 2>/dev/null; kill ${SAMP:-} 2>/dev/null; setcap 200; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16
LD_LIBRARY_PATH=$LP/lib $LP/bin/llama-server -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -fa on -np 16 -cb -c 524288 -b 2048 -ub 2048 --host 127.0.0.1 --port $PORT > $B/$TAG-server.log 2>&1 &
S=$!
if wait_server $PORT $S; then log "server up"; python3 $B/powercap-adaptive.py http://127.0.0.1:$PORT $TAG --conc 16 --coarse 200,185,170,155,140,125,110,95,80 --pairs-coarse 1 --pairs-fine-min 2 --pairs-fine-max 4 > $B/$TAG-controller.out 2>&1; rc=$?; log "controller rc=$rc"
else log "server FAILED"; rc=1; fi
stop_server $S; setcap 200
sleep 5; bad=0; for d in $DIES; do m=$(grep '\*' /sys/bus/pci/devices/0000:$d:00.0/pp_dpm_sclk | awk '{print $2}'); [ "$(tr -dc 0-9 <<<"$m")" -ge 1700 ] || bad=1; done; log "final idle check: $([ $bad = 0 ] && echo OK || echo CLAMPED)"
kill $SAMP 2>/dev/null; restore; log "ALLDONE rc=$rc"; [ $bad = 0 ] || { touch $B/.clamp-detected; exit 2; }; touch $B/.$TAG-done; exit $rc
