#!/bin/bash
# Fan power delta, GPUs IDLE (lead, 2026-09-19: "fans use a lot of power that typically isn't
# accounted for"). 90 s at max RPM, 90 s on the production PWM curve, same idle load.
# The difference is what every max-fan DC total in the corpus has been spending on cooling.
set -u
W=/root/night-20260919; B=/root/rocm-tests/bench
SMC=/sys/devices/LNXSYSTM:00/LNXSYBUS:00/PNP0A08:00/device:1f/APP0001:00
out=$W/fanpower.md
log() { echo "$(date -Is) [fanpower] $*" | tee -a $W/fanpower.progress; }
sample() { local label=$1; local secs=$2; local f=$W/fanpower-$label.log; : > $f
  # NB: the three assignments MUST be separate `local` statements. Bash expands every word in a
  # single `local a=$1 b=$2 c=$a` BEFORE performing any assignment, so $label was empty and both
  # samples wrote to the same truncated file. That was the whole bug.
  nohup $B/smc-log.sh $f >/dev/null 2>&1 & local sp=$!
  sleep $secs; kill $sp 2>/dev/null
  local dc rpm
  dc=$(sed -n 's/.*PZ0G=\([0-9.]*\).*/\1/p' $f | sort -n | awk '{a[NR]=$1} END{print a[int(NR/2)+1]}')
  rpm=$(cat $SMC/fan1_input)/$(cat $SMC/fan2_input)/$(cat $SMC/fan3_input)/$(cat $SMC/fan4_input)
  if [ -z "$dc" ]; then
    echo "FATAL: sample '$label' produced no PZ0G. smc-read.py broken or log empty ($f)." >&2
    echo "$label|FAILED|$rpm"; return 1
  fi
  echo "$label|$dc|$rpm"; }
restore_fans() { cp /root/t2fand.conf.orig /etc/t2fand.conf; systemctl restart t2fanrd; }
log "=== start (GPUs idle)"
sed -i 's/always_full_speed=false/always_full_speed=true/' /etc/t2fand.conf; systemctl restart t2fanrd; sleep 25
MAXR=$(sample maxfans 90); log "max fans: $MAXR"
cp /root/t2fand.conf.orig /etc/t2fand.conf; systemctl restart t2fanrd; sleep 45
PWMR=$(sample pwmfans 90); log "pwm fans: $PWMR"
case "$MAXR$PWMR" in *FAILED*) log "ABORT: a sample failed; not writing $out"; restore_fans; exit 1;; esac
{ echo "# Fan power at idle — what max-RPM testing has been spending on cooling   $(date -Is)"; echo
  echo "GPUs idle, host idle, same state otherwise. Median SMC DC total (PZ0G) over 90 s."; echo
  echo "| fan mode | DC total W | fan RPM (1/2/3/4) |"; echo "|---|---:|---|"
  echo "| max (all corpus testing) | $(echo $MAXR|cut -d'|' -f2) | $(echo $MAXR|cut -d'|' -f3) |"
  echo "| production PWM curve | $(echo $PWMR|cut -d'|' -f2) | $(echo $PWMR|cut -d'|' -f3) |"
  echo
  awk -v a="$(echo $MAXR|cut -d'|' -f2)" -v b="$(echo $PWMR|cut -d'|' -f2)" \
    'BEGIN{printf "**Fan power delta at idle: %.1f W** of the DC total.\n", a-b}'
  echo; echo "Note: at idle the PWM curve sits near its floor, so this is close to the MAXIMUM"
  echo "delta. Under serving load the production curve ramps and the gap narrows."; } > $out
log "=== done: $out"; touch $W/.fanpower-done
