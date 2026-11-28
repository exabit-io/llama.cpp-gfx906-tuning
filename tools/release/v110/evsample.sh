#!/bin/bash
# evsample.sh OUT [STOPFILE] -- passive sampler of KFD queue evictions, once a second, until STOPFILE exists.
# For every process KFD knows (/sys/class/kfd/kfd/proc/<pid>) one line per second while it lives:
#   epoch-seconds  HH:MM:SS  pid  comm  evicted_ms (sum over the process's stats_<gpu>/evicted_ms)
# evicted_ms is the time the process's GPU queues were evicted (unmapped by the driver until its buffers are
# validated again). Four small sysfs reads a second: no load. Reads only; changes nothing.
set -u
OUT=${1:?OUT file}; STOP=${2:-$OUT.stop}
rm -f "$STOP"
echo "# $(date -Is) evsample start (pid $$)" >> "$OUT"
while [ ! -e "$STOP" ]; do
  now=$(date +%s); hms=$(date -u +%H:%M:%S)
  for d in /sys/class/kfd/kfd/proc/[0-9]*; do
    [ -d "$d" ] || continue
    pid=${d##*/}; s=0
    for f in "$d"/stats_*/evicted_ms; do
      [ -r "$f" ] || continue
      v=$(cat "$f" 2>/dev/null); s=$(( s + ${v:-0} ))
    done
    comm=$(cat /proc/"$pid"/comm 2>/dev/null); echo "$now $hms $pid ${comm:-?} $s" >> "$OUT"
  done
  command sleep 1
done
echo "# $(date -Is) evsample end" >> "$OUT"
