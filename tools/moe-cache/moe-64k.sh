#!/bin/bash
# moe-64k.sh [TAG] -- DeepSeek-V4.1-Flash UD-IQ1_S (217 GiB of files, 110 GiB of experts, 41 layers: larger than the
# 128 GB of VRAM) at ONE stream x 65536 tokens, with the experts in VRAM as far as they fit against every expert in
# host RAM with and without upstream's MoE cache. Installed package 1.1.0, llama-batched-bench, f16 KV, -b 2048
# -ub 2048, 125 W per die, 1,024 generated tokens.
#
# Why (lead, 2026-11-28): 1.5 TB of RAM per machine for bigger MoE models whose experts live in DRAM. This model is
# the stand-in on disk: it can be run both ways today. moe-cache-check.sh (functional, 4K): the cache works in layer
# split (not in tensor split, by its source), 79 % hit rate after 24 tokens; all experts in RAM 7.9 tok/s without the
# cache, 12.2 with 32 GiB; as much as fits in VRAM 24 (layer split) and 30 (tensor split). One 18-token prompt batch
# uploaded 13 GiB of experts: large batches upload what they use, the cache serves ubatches of at most 32 tokens.
#
# Arms, in this order, ONE run each (a first look):
#   lfit   -sm layer, the tree's automatic placement (what fits goes to VRAM)
#   tfit   -sm tensor, the fork's placement (the fastest known way for this model; the cache cannot be used)
#   c64    -sm layer -cmoe --moe-cache-mib 65536        every expert in RAM, 64 GiB of them cached in VRAM
#   c32    -sm layer -cmoe --moe-cache-mib 32768
#   c64s   c64 with --moe-cache-cpu-share 0.5           half of the misses computed on the CPU
#   cmoe   -sm layer -cmoe                              every expert in RAM, no cache: the CPU computes them
#
# READING, WRITTEN BEFORE THE FIRST RUN. n=1 per arm: a first look; differences under 10 % are not read.
#   (L1) Per arm: prefill tok/s at 64K, decode tok/s over 1,024 tokens at 64K depth (the cache warms inside the run),
#        the cache's hit rate and uploaded GiB (its own log line), peak VRAM per die, DC and CPU-zone power.
#   (L2) The question for the RAM plan, on the best arm with every expert in RAM (c64, c32, c64s, cmoe):
#        (a) decode at 64K at least 12 tok/s on one stream (R3.1's floor per request), and
#        (b) prefill at least 110 tok/s (a 64K prompt in under ten minutes).
#        Either "no": a model whose experts must live in RAM does not reach the service's floor on this box however
#        much RAM is installed; that is then said plainly, with the numbers. Both "yes": n=3 and four slots are next.
#   (L3) What having the experts in VRAM is worth: lfit and tfit against the best all-in-RAM arm, as a ratio.
#   (L4) An arm that fails (does not fit, aborts, runs over 45 minutes) is reported with its log lines.
set -u
W=/root/moe-cache; R=/root/rocm-tests/bench; TAG=${1:-m1}; D=65536; S=1; NTG=1024
TL=/root/llama.cpp-benchmarking/tools; CM=$TL/cell-metrics.py
PKG=/opt/mx-llama.cpp; M=/root/models/DeepSeek-V4.1-Flash-1017-UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf
D4="--device rocm0,rocm1,rocm2,rocm3"
ARMS=${ARMS:-"lfit tfit c64 c32 c64s cmoe"}
P=$W/moe64-$TAG.progress; DONE=$W/.moe64-$TAG-done; O=$W/moe64-$TAG; RUNS=$W/moe64-$TAG.tsv
log(){ echo "$(date -Is) [moe64] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
args(){ case $1 in lfit) echo "-sm layer";; tfit) echo "-sm tensor";; c64) echo "-sm layer -cmoe --moe-cache-mib 65536 -lv 4";;
  c32) echo "-sm layer -cmoe --moe-cache-mib 32768 -lv 4";; c64s) echo "-sm layer -cmoe --moe-cache-mib 65536 --moe-cache-cpu-share 0.5 -lv 4";;
  cmoe) echo "-sm layer -cmoe";; esac; }
if [ "${DRY:-0}" = 1 ]; then for a in $ARMS; do echo "$a: $(args $a)"; done; exit 0; fi
for p in llama-server llama-batched-b llama-perplexit llama-completi test-backend-op llama-cli cmake ninja; do
  pgrep -x $p >/dev/null && { log "FATAL: $p is running"; exit 1; }
done
[ -e $RUNS ] && { log "FATAL: result file of tag $TAG exists"; exit 1; }
mkdir -p $O; rm -f $DONE
log "package $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906); arms: $ARMS; RAM free $(awk '/MemAvailable/{printf "%.0f GiB", $2/1048576}' /proc/meminfo); load $(cut -d' ' -f1-3 /proc/loadavg)"
. $R/gpu-test-env.sh
export GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1
BPID=""; VRS=""
cleanup(){ trap - INT TERM EXIT; [ -n "$BPID" ] && { pkill -TERM -P "$BPID" 2>/dev/null; kpid "$BPID"; }; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; kpid "${VRS:-}"; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/moe64-$TAG-clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/moe64-$TAG-smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $W/moe64-$TAG-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/moe64-$TAG-smc.log >/dev/null 2>&1 &
SMCL=$!
( while :; do echo "$(date +%s) $(cat /sys/class/drm/card[0-9]/device/mem_info_vram_used | tr '\n' ' ')"; sleep 5; done ) >> $W/moe64-$TAG-vram.log 2>/dev/null &
VRS=$!
sleep 6
log "$(state_line)"
for arm in $ARMS; do
  out=$O/$arm.md; t0=$(date +%s)
  ( exec env -u LD_LIBRARY_PATH timeout 2700 $PKG/bin/llama-batched-bench -m $M $D4 $(args $arm) \
      -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $(( S*(D+1280) )) -npp $D -ntg $NTG -npl $S > $out 2>${out%.md}.log < /dev/null ) &
  BPID=$!; wait $BPID; rc=$?; BPID=""; t1=$(date +%s)
  peak=$(awk -v a=$t0 -v b=$t1 '$1 >= a && $1 <= b { n++; for (i = 2; i <= 5; i++) if ($i > m[i]) m[i] = $i } END { if (n) printf "%.1f %.1f %.1f %.1f", m[2]/2^30, m[3]/2^30, m[4]/2^30, m[5]/2^30; else printf "n/a" }' $W/moe64-$TAG-vram.log)
  cache=$(grep -hE 'llama_moe_cache: ubatch|MoE cache size' ${out%.md}.log | sed 's/^[0-9.]* [A-Z] //; s/^impl: *//' | tr '\n' ';' | cut -c1-700)
  if m=$(python3 -I $CM "$out" $D $S 2>>$P); then
    printf "%s\t%s\t%s\t%s\t%s\n" "$arm" "$t0" "$t1" "$rc" "$m" >> $RUNS
    log "$arm ($(( t1 - t0 )) s): $(echo "$m" | awk -F'\t' '{printf "decode %.3f tok/s, prefill %.1f tok/s",$1,$2}'); peak VRAM GiB $peak"
  else
    log "$arm ($(( t1 - t0 )) s): NO RESULT rc=$rc; peak VRAM GiB $peak; $(grep -v 'New LWP\|^\[Thread\|^warning:' ${out%.md}.log | grep -iE 'error|abort|failed|out of memory|alloc|not support' | tail -3 | tr '\n' ' ' | cut -c1-400)"
  fi
  [ -n "$cache" ] && log "$arm cache: $cache"
done
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${SMCL:-}"; kpid "${VRS:-}"; touch $DONE; sleep 6; kpid "${WDOG:-}"; restore
python3 -I - $W $TAG <<'PY' | tee -a $P
import datetime as dt, re, sys
w, tag = sys.argv[1:3]
smc = []; day = None; last = None
for l in open('%s/moe64-%s-smc.log' % (w, tag)):
    if l.startswith('#'):
        m = re.match(r'# (\S+) smc-log start', l)
        if m: day = dt.datetime.fromisoformat(m.group(1)).date(); last = None
        continue
    f = l.split()
    try:
        t = dt.datetime.combine(day, dt.time.fromisoformat(f[0]), tzinfo=dt.timezone.utc)
        kv = dict(x.split('=') for x in f[1:] if '=' in x); v = (float(kv['PZ0G']), float(kv['PZ1G']))
    except Exception: continue
    if last is not None and t < last - dt.timedelta(hours=12): day += dt.timedelta(days=1); t += dt.timedelta(days=1)
    last = t; smc.append((t.timestamp(), v))
rows = []
for l in open('%s/moe64-%s.tsv' % (w, tag)):
    f = l.rstrip('\n').split('\t')
    arm, t0, t1, dec, pre = f[0], int(f[1]), int(f[2]), float(f[4]), float(f[5])
    tdec = 1024 / dec
    d = [v for (t, v) in smc if t1 - 2 - tdec <= t <= t1 - 2]; p = [v for (t, v) in smc if t0 + 60 <= t <= t1 - 2 - tdec]
    mean = lambda x, i: sum(v[i] for v in x) / len(x) if x else float('nan')
    rows.append((arm, dec, pre, 65536 / pre / 60, mean(d, 0), mean(d, 1), mean(p, 0), mean(p, 1)))
print('[moe64] arm | decode tok/s at 64K | prefill tok/s | minutes for a 64K prompt | DC W / CPU-zone W in decode | DC W / CPU-zone W in prefill')
for r in rows: print('[moe64]   %-5s | %7.3f | %7.1f | %5.1f | %4.0f / %3.0f | %4.0f / %3.0f' % r)
ram = [r for r in rows if r[0] in ('c64', 'c32', 'c64s', 'cmoe')]; fit = [r for r in rows if r[0] in ('lfit', 'tfit')]
if ram:
    bd = max(ram, key=lambda r: r[1]); bp = max(ram, key=lambda r: r[2])
    print('[moe64] (L2a) best decode with every expert in RAM: %s %.2f tok/s -> %s 12 tok/s' % (bd[0], bd[1], 'AT OR OVER' if bd[1] >= 12 else 'UNDER'))
    print('[moe64] (L2b) best prefill with every expert in RAM: %s %.1f tok/s (%.1f min for 64K) -> %s 110 tok/s' % (bp[0], bp[2], bp[3], 'AT OR OVER' if bp[2] >= 110 else 'UNDER'))
    for r in fit:
        print('[moe64] (L3) %s against the best all-in-RAM arm: decode %.2fx, prefill %.2fx' % (r[0], r[1] / bd[1], r[2] / bp[2]))
PY
log "=== MOE-64K $TAG DONE ==="
