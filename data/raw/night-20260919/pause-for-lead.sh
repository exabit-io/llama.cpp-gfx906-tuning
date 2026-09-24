#!/bin/bash
# pause-for-lead.sh — hand the GPUs back cleanly, between cells.
#
# Ctrl-Z on the bench would NOT work: a stopped llama-batched-bench keeps ~31 GiB per die allocated,
# so the box would look busy and the lead's test would find no memory. It would also corrupt the
# in-flight cell, whose metric is wall-clock based.
#
# Instead: SIGSTOP the PARENT script while it waits on its child. The current cell runs to completion
# (valid data), its process exits and releases all VRAM, and no new cell starts.
set -u
W=/root/night-20260919
log(){ echo "$(date -Is) [pause] $*" | tee -a $W/pause.log; }
# PIDs are ARGUMENTS, not pattern matches: pgrep -f on a script name matches any shell quoting it,
# including this one, so a pattern-driven `kill -STOP` can suspend the wrong process. Trap T4.
S=${1:?usage: pause-for-lead.sh RUN_PID [GUARD_PID...]}
shift; GUARDS="$*"
kill -0 "$S" 2>/dev/null || { log "pid $S is not running — nothing to pause"; exit 1; }
log "suspending the run script (pid $S) — the cell in flight will finish first"
kill -STOP "$S"
# suspend the guards too: the watchdog's breach path runs pkill -f llama-*, which would hit the
# lead's processes, and the sampler would keep writing during someone else's test.
log "waiting for the in-flight cell to finish (up to ~7 min)"
for i in $(seq 1 120); do
  n=$(pgrep -cf '^/root/build-[^ ]*/bin/llama-batched-bench' || true)
  [ "${n:-0}" -eq 0 ] && break
  sleep 5
done
n=$(pgrep -cf '^/root/build-[^ ]*/bin/llama-batched-bench' || true)
if [ "${n:-0}" -gt 0 ]; then
  log "cell STILL running after 10 min — leaving it alone rather than corrupting it; re-run this script"
  exit 2
fi
# guards come down only NOW: suspending them while a cell is still loaded left minutes of four
# pinned dies with no 1228 W envelope guard (noted 2026-09-20).
for p in $GUARDS; do kill -STOP "$p" 2>/dev/null && log "suspended guard pid $p"; done
log "cell finished. cells recorded so far: $(wc -l < $W/d1conf.tsv 2>/dev/null || echo 0)/40"
# restore the box to the lead's conditions
for d in 0b 0e 1b 1e; do echo 200000000 > /sys/bus/pci/devices/0000:$d:00.0/hwmon/hwmon*/power1_cap 2>/dev/null; done
RAPL=/sys/class/powercap/intel-rapl:0
MAXW=$(cat $RAPL/constraint_0_max_power_uw 2>/dev/null || echo "")
if [ -n "$MAXW" ] && [ "$MAXW" -gt 0 ]; then
  for c in 0 1; do echo "$MAXW" > $RAPL/constraint_${c}_power_limit_uw 2>/dev/null; done
  log "host RAPL restored to $((MAXW/1000000)) W (was capped at 150 W by the harness)"
else
  log "could not read the RAPL maximum — host is STILL CAPPED at 150 W, tell the lead"
fi
cp /root/t2fand.conf.orig /etc/t2fand.conf 2>/dev/null && systemctl restart t2fanrd && log "fans back on the PWM curve"
log "VRAM per die now: $(rocm-smi --showmemuse 2>/dev/null | grep -oE '[0-9]+ *%' | tr '\n' ' ')"
log "=== GPUs ARE FREE. resume with: /root/night-20260919/resume-after-lead.sh"
