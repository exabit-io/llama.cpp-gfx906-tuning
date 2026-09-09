#!/bin/bash
# HBM bandwidth vs per-die power cap (2026-09-08 00:45, user question: minimum wattage for full HBM2 bandwidth). All four dies run
# hbm-bw (1 GiB float4 copy + read-only reduction, 8 s, median GB/s) concurrently at each cap; sclk and die power sampled mid-run.
# mclk/socclk/fclk have no DPM on this firmware (1000/971/1166 MHz fixed), so the cap can only throttle sclk. ~4 min GPU.
set -u
B=/root/rocm-tests/bench; TAG=hbm-bw-cap-sweep; . $B/gpu-test-env.sh
DIES="0b 0e 1b 1e"; capfile() { ls /sys/bus/pci/devices/0000:$1:00.0/hwmon/hwmon*/power1_cap; }
setcap() { local w=$1 d; for d in $DIES; do echo $((w*1000000)) > $(capfile $d); done; }
PROG=$B/$TAG.progress; : > $PROG; OUT=$B/$TAG.md; log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; setcap 200; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
CAPS="200 170 140 120 110 100 95 90 85 80 70 60 50 200"
echo "# $TAG $(date -Is)  HBM bandwidth vs per-die cap, four dies concurrently, hbm-bw 1 GiB, perf level high, 8 s per die per cap (median GB/s)" > $OUT
echo "" >> $OUT; echo "| cap W | die | copy GB/s | read GB/s | sclk mid-run | die W mid-run |" >> $OUT; echo "|---:|---|---:|---:|---:|---:|" >> $OUT
for w in $CAPS; do
  setcap $w; sleep 8
  i=0; pids=""; for d in $DIES; do $B/hbm-bw $i 8 > $B/$TAG-$w-$d.out 2>&1 & pids="$pids $!"; i=$((i+1)); done
  sleep 4; mid=""; for d in $DIES; do dev=/sys/bus/pci/devices/0000:$d:00.0; mid="$mid $d:$(grep '\*' $dev/pp_dpm_sclk | awk '{print $2}'):$(awk '{printf "%dW",$1/1e6}' $dev/hwmon/hwmon*/power1_input | head -1)"; done
  wait $pids   # only the four probes; a bare wait also waits for the clock sampler subshell and hangs (2026-09-08 02:27 run)
  line="CAP $w:"; for d in $DIES; do r=$(cat $B/$TAG-$w-$d.out); pci=$(awk '{print $4}' <<<"$r"); c=$(sed -n 's/.*copy_GBs \([0-9.]*\).*/\1/p' <<<"$r"); rd=$(sed -n 's/.*read_GBs \([0-9.]*\).*/\1/p' <<<"$r")
    # map by PCI bus id printed by the tool, not by launch index
    pd=$(sed -n 's/^0000:\([0-9a-f][0-9a-f]\):00\.0$/\1/p' <<<"$pci"); m=$(tr ' ' '\n' <<<"$mid" | grep "^$pd:" | cut -d: -f2-); s=${m%%:*}; pw=${m##*:}
    line="$line  $pd copy ${c:-ERR} read ${rd:-ERR} sclk ${s:-?} ${pw:-?}"; echo "| $w | $pd | ${c:-ERR} | ${rd:-ERR} | ${s:-?} | ${pw:-?} |" >> $OUT
    [ -z "$c" ] && { echo "| $w | $pd | ERR: $r |" >> $OUT; }
  done; log "$line"
done
setcap 200; kill $SAMP 2>/dev/null; restore; log "ALLDONE"; exit 0
