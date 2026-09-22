#!/bin/bash
# resume-after-lead.sh — put the campaign conditions back and continue where it stopped.
set -u
W=/root/night-20260919
log(){ echo "$(date -Is) [resume] $*" | tee -a $W/pause.log; }
busy=$(pgrep -cf '^/root/build-[^ ]*/bin/llama-' || true)
if [ "${busy:-0}" -gt 0 ]; then log "REFUSING: something is still using the GPUs"; exit 3; fi
# campaign conditions: 125 W per die, fans pinned, host capped. Every cell in this campaign was
# measured this way; resuming under different conditions would make the remaining cells incomparable.
sed -i 's/always_full_speed=false/always_full_speed=true/' /etc/t2fand.conf; systemctl restart t2fanrd
rocm-smi --setperflevel high >/dev/null 2>&1
for c in 0 1; do echo 150000000 > /sys/class/powercap/intel-rapl:0/constraint_${c}_power_limit_uw 2>/dev/null; done
for d in 0b 0e 1b 1e; do echo 125000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
sleep 8
log "conditions restored: 125 W/die, fans max, host RAPL 150 W"
for p in $GUARDS; do kill -CONT "$p" 2>/dev/null && log "resumed guard pid $p"; done
S=${1:?usage: resume-after-lead.sh RUN_PID [GUARD_PID...]}
shift; GUARDS="$*"
kill -0 "$S" 2>/dev/null || { log "pid $S is GONE — the run must be restarted, not resumed"; exit 1; }
kill -CONT "$S" && log "resumed run script pid $S; cells so far $(wc -l < $W/d1conf.tsv 2>/dev/null || echo 0)/40"
log "NOTE FOR THE ANALYSIS: a pause happened mid-run. Cells are independent, but arms are interleaved"
log "to cancel drift. If the dies were heated by the interleaved test, the rep spanning the pause"
log "should be discarded and repeated rather than trusted."
