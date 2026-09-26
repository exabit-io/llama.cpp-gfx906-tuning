#!/bin/bash
# r2mon.sh — Monitor source for round 2 (chain-r2c -> binrun r2 -> fncompat r2); offsets persist in .r2mon.state (re-arm safe)
W=/root/night-20260919; ST=$W/.r2mon.state
declare -A off; [ -f $ST ] && . $ST
FILES="chain-r2.log binrun-r2.progress fncompat-r2.progress binrun-r2-watchdog.out fncompat-r2-watchdog.out"
PAT='PASS|FAIL|DONE|STOPPED|FATAL|ready|FAILED|COMPARABLE|comparable|GATE|gate passed|b[1-9]: decode|FALLBACK|=== axis|CLAMP|kill|ENVELOPE|refusing|not measured|no build|skipped|starting|chain-r2'
alive(){ [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
while true; do
  for f in $FILES; do
    [ -f $W/$f ] || continue
    k=${f//[.-]/_}; o=${off[$k]:-0}; n=$(wc -l < $W/$f)
    if [ "$n" -gt "$o" ]; then sed -n "$((o+1)),${n}p" $W/$f | grep -E "$PAT" | sed "s/^/[$f] /" | cut -c1-400; off[$k]=$n; fi
  done
  declare -p off > $ST
  any=0
  for x in "chain-r2:" "binrun-r2:.binrun-r2-done" "fncompat-r2:.fncompat-r2-done"; do
    nm=${x%%:*}; fl=${x#*:}; [ -f $W/$nm.pid ] || continue; p=$(cat $W/$nm.pid)
    if alive "$p"; then any=1
    elif [ -n "$fl" ] && [ ! -f $W/$fl ] && ! grep -q "^$nm$" $ST.dead 2>/dev/null; then echo "$nm pid $p EXITED WITHOUT done flag"; echo $nm >> $ST.dead; fi
  done
  [ $any = 0 ] && { echo "round-2 chain not running (finished, or not started yet)"; exit 0; }
  sleep 10
done
