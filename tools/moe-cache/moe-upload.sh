#!/bin/bash
# moe-upload.sh [TAG] -- DIAGNOSTIC, not a result (8K prompt: under the 64K floor): does the way the model file is
# loaded change the prefill rate when every expert is in host RAM?
#
# Why: moe-64k.sh arm c64 (every expert in RAM, 64 GiB cache) read 15.95 tok/s decode at 64K with a 96.9 % hit rate
# and 38.5 tok/s PREFILL (a 64K prompt in 28 minutes). During that prefill the machine drew 522 W DC (dense prefill:
# 830 W) and the CPU zone at most 95 W: neither the dies nor the CPU were busy, so the time goes into moving experts
# from host memory to the dies. A 2048-token batch took 53 s; if it uploads about all 110 GiB of experts that is
# about 2 GiB/s, a sixth of what a PCIe 3.0 x16 link carries. By default the file is mmap-ed (page cache).
#
# Arms (DeepSeek-V4.1-Flash, -sm layer -cmoe --moe-cache-mib 65536, one stream x 8192 tokens, 64 generated), once:
#   mmap    the default load
#   dio     -lm dio    (direct I/O into the process's own memory)
#   mlock   -lm mlock  (page-locked)
#   none    -lm none   (read into the process's memory, no mapping)
#
# READING, WRITTEN BEFORE THE RUN.
#   (U1) Prefill tok/s per arm. An arm at least 1.5x the default is a lever: it is then run once at 64K
#        (ARMS=<arm> D=65536 NTG=1024) and that number replaces c64's in the verdict on the RAM plan.
#   (U3) Added after u1 (all four load modes 89 to 94 tok/s at 8K against 38.5 at 64K, so the rate falls with depth and
#        the load mode is not the limit; in the 64K run die 0 was 81 % busy and the others about 20 %): arms lfit
#        (-sm layer, automatic placement) and tfit (-sm tensor) at the same 8K give the prefill rate with the experts in
#        VRAM. Written before running them: if they are within 1.5x of the 94 tok/s, having the experts in RAM is not
#        what makes this model's prefill slow; if they are faster by more, the difference is what RAM placement costs.
#   (U2) No arm 1.5x faster: the load mode is not the limit; the copy path itself is (a code question:
#        pinned staging for expert uploads), and the verdict stands at 38.5 tok/s.
set -u
W=/root/moe-cache; R=/root/rocm-tests/bench; TAG=${1:-u1}; D=${D:-8192}; S=1; NTG=${NTG:-64}
TL=/root/llama.cpp-benchmarking/tools; CM=$TL/cell-metrics.py
PKG=/opt/mx-llama.cpp; M=/root/models/DeepSeek-V4.1-Flash-1017-UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf
D4="--device rocm0,rocm1,rocm2,rocm3"
ARMS=${ARMS:-"mmap none dio mlock"}
P=$W/moeup-$TAG.progress; DONE=$W/.moeup-$TAG-done; O=$W/moeup-$TAG; RUNS=$W/moeup-$TAG.tsv
log(){ echo "$(date -Is) [moeup] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
args(){ case $1 in mmap) echo "-sm layer -cmoe --moe-cache-mib 65536";; dio) echo "-sm layer -cmoe --moe-cache-mib 65536 -lm dio";;
  mlock) echo "-sm layer -cmoe --moe-cache-mib 65536 -lm mlock";; none) echo "-sm layer -cmoe --moe-cache-mib 65536 -lm none";;
  lfit) echo "-sm layer";; tfit) echo "-sm tensor";; esac; }
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
setmax; start_sampler $W/moeup-$TAG-clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/moeup-$TAG-smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $W/moeup-$TAG-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/moeup-$TAG-smc.log >/dev/null 2>&1 &
SMCL=$!
( while :; do echo "$(date +%s) $(cat /sys/class/drm/card[0-9]/device/mem_info_vram_used | tr '\n' ' ')"; sleep 5; done ) >> $W/moeup-$TAG-vram.log 2>/dev/null &
VRS=$!
sleep 6
log "$(state_line)"
for arm in $ARMS; do
  out=$O/$arm.md; t0=$(date +%s)
  ( exec env -u LD_LIBRARY_PATH timeout 2700 $PKG/bin/llama-batched-bench -m $M $D4 $(args $arm) \
      -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $(( S*(D+1280) )) -npp $D -ntg $NTG -npl $S > $out 2>${out%.md}.log < /dev/null ) &
  BPID=$!; wait $BPID; rc=$?; BPID=""; t1=$(date +%s)
  peak=$(awk -v a=$t0 -v b=$t1 '$1 >= a && $1 <= b { n++; for (i = 2; i <= 5; i++) if ($i > m[i]) m[i] = $i } END { if (n) printf "%.1f %.1f %.1f %.1f", m[2]/2^30, m[3]/2^30, m[4]/2^30, m[5]/2^30; else printf "n/a" }' $W/moeup-$TAG-vram.log)
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
for l in open('%s/moeup-%s-smc.log' % (w, tag)):
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
for l in open('%s/moeup-%s.tsv' % (w, tag)):
    f = l.rstrip('\n').split('\t')
    arm, t0, t1, dec, pre = f[0], int(f[1]), int(f[2]), float(f[4]), float(f[5])
    tdec = 64 / dec
    d = [v for (t, v) in smc if t1 - 2 - tdec <= t <= t1 - 2]; p = [v for (t, v) in smc if t0 + 60 <= t <= t1 - 2 - tdec]
    mean = lambda x, i: sum(v[i] for v in x) / len(x) if x else float('nan')
    rows.append((arm, dec, pre, 65536 / pre / 60, mean(d, 0), mean(d, 1), mean(p, 0), mean(p, 1)))
print('[moeup] arm | decode tok/s at 64K | prefill tok/s | minutes for a 64K prompt | DC W / CPU-zone W in decode | DC W / CPU-zone W in prefill')
for r in rows: print('[moeup]   %-5s | %7.3f | %7.1f | %5.1f | %4.0f / %3.0f | %4.0f / %3.0f' % r)
base = [r for r in rows if r[0] == 'mmap']
if base:
    for r in rows:
        if r[0] != 'mmap': print('[moeup] (U1) %s prefill against the default load: %.2fx -> %s' % (r[0], r[2] / base[0][2], 'A LEVER (at least 1.5x)' if r[2] >= 1.5 * base[0][2] else 'not a lever'))
PY
log "=== MOE-UPLOAD $TAG DONE ==="
