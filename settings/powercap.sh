#!/usr/bin/env bash
# powercap.sh - set the per-die power cap on all four gfx906 dies (watts), or show it.
#
#   settings/powercap.sh            show caps, sclk and power per die
#   settings/powercap.sh 125        set every die to 125 W
#   settings/powercap.sh 200        back to the board maximum
#
# Measured on the production build at 16 clients (run-through, power cap study):
#   cap W   agg tok/s   vs 200   gen tok/kJ (bays)
#   200     87.1        +0%      105          reference
#   185     86.6        -0.6%    108          "within 2%"
#   170     84.7        -2.7%    113          "within 5%"
#   140     79.4        -8.9%    124          "within 10%"
#   125     75.3        -13.5%   131          production cap on the hyperconverged nodes (1228 W envelope + marginal economics)
#   85      60.6        -30%     146          the DPM floor (sclk 999 MHz, ~83 W/die under load): any lower value is accepted and ignored
# HBM2 bandwidth (880-892 GB/s read) does not move with the cap. Fill the slots before lowering the cap: 16 clients at 200 W
# is as efficient per kJ as 8 clients at the floor.
set -euo pipefail
show() {
  for d in /sys/class/drm/card*/device; do
    [ -e "$d/hwmon" ] || continue
    h=$(ls -d "$d"/hwmon/hwmon* | head -1)
    [ -e "$h/power1_cap" ] || continue
    cap=$(( $(cat "$h/power1_cap") / 1000000 )); pw=$(( $(cat "$h/power1_average" 2>/dev/null || echo 0) / 1000000 ))
    sclk=$(grep '\*' "$d/pp_dpm_sclk" 2>/dev/null | awk '{print $2}')
    echo "$(basename "$(dirname "$d")")  $(cat "$d/uevent" | grep PCI_SLOT_NAME | cut -d= -f2)  cap ${cap} W  power ${pw} W  sclk ${sclk:-?}"
  done
}
if [ $# -eq 0 ]; then show; exit 0; fi
W=$1
if [ "$W" -lt 85 ]; then echo "note: caps below ~85 W are not honoured by the firmware (DPM floor); setting anyway" >&2; fi
for d in /sys/class/drm/card*/device; do
  [ -e "$d/hwmon" ] || continue
  h=$(ls -d "$d"/hwmon/hwmon* | head -1)
  [ -e "$h/power1_cap" ] || continue
  echo $(( W * 1000000 )) > "$h/power1_cap"
done
show
