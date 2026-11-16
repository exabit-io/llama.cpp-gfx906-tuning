#!/bin/bash
# pkg-resid.sh -- is any of the -r1 package's 4 x 64K decode difference its own? rocm-ab3.sh measured the installed
# package (master 290fc642a compiled by ROCm 10.1's clang 24 with the tile rule, no global host flag) at 24.737 tok/s per
# slot against 24.868 for /root/build-master-v100r on the 10.0 runtime (-0.52 %, n = 5 each). The runtime's share was
# measured in other sessions at 0.32 % (rt-swap) and 0.6 % (pkg-swap), so the remainder is not known from one session.
# Method and arms are pkg-swap.sh's (all libraries preloaded by full path, loaded set read back from /proc/<pid>/maps):
#   r100   the reference build on the ROCm 10.0 runtime (rocm-ab3.sh's reference arm)
#   all    the package's program with all of the reference build's libraries, on the ROCm 10.1 runtime
#   pkg    the package as installed
#   hip    the package with the reference build's libggml-hip        (stage 2)
#   llama  the package with the reference build's libllama, libllama-common, bench library   (stage 2)
#   ggml   the package with the reference build's libggml-base, libggml-cpu, libggml          (stage 2)
# READING, WRITTEN BEFORE THE FIRST CELL. Decode tok/s per slot at 4 x 64K, N blocks (default 5), seeded random order.
# Stage 1 (ARMS="r100 all pkg"): r100 - all is the runtime's share; all - pkg is the package's own share. The package has a
# share of its own only if every pkg sample is below every "all" sample AND mean(pkg) / mean(all) - 1 <= -0.10 %.
# Otherwise the item is closed: the package's decode difference from the ROCm 10.0 source build is the runtime's.
# Stage 2 only if stage 1 finds a share (ARMS="all pkg hip llama ggml"): a group carries it if every sample of its swap
# arm is above every pkg sample; if no group does, it is spread or below this method's resolution, and is reported so.
# Prefill is listed, not read (the arms' GPU code differs only in the tile rule, measured in gpu-opt-confirm.sh).
set -u
W=/root/rocm101-migration-20261114; R=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf
TL=/root/llama.cpp-benchmarking/tools; CM=$TL/cell-metrics.py; NF=$TL/assert-no-fa-fallback.sh
N=${N:-5}; ARMS=${ARMS:-"r100 all pkg"}; TAG=${TAG:-pkg-resid}
P=$W/$TAG.progress; TSV=$W/$TAG.tsv; DONE=$W/.$TAG-done; O=$W/$TAG; mkdir -p $O
log(){ echo "$(date -Is) [swap] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
rm -f $DONE; echo $$ > $W/$TAG.pid
PK=/opt/mx-llama.cpp/lib; RF=/root/build-master-v100r/bin; RT=/opt/rocm/core-10.1; ALT=${ALT:-/root/build-master-v100rn/bin}
LIBS="libggml-base.so.1 libggml.so.1 libggml-cpu.so.1 libggml-hip.so.1 libllama.so.1 libllama-common.so.1 libllama-batched-bench-impl.so"
from_ref(){ # from_ref ARM LIB -> 0 if LIB comes from the reference build in ARM
  case "$1:$2" in
    all:*) return 0;;
    llama:libllama.so.1|llama:libllama-common.so.1|llama:libllama-batched-bench-impl.so) return 0;;
    llaman:libllama.so.1|llaman:libllama-common.so.1|llaman:libllama-batched-bench-impl.so) return 0;;
    ggml:libggml-base.so.1|ggml:libggml-cpu.so.1|ggml:libggml.so.1) return 0;;
    hip:libggml-hip.so.1) return 0;;
  esac
  return 1
}
want(){ # want ARM -> the eight (seven sonames) full paths of the arm, one per line, resolved
  local a=$1 l
  for l in $LIBS; do
    if [ "$a" = llaman ] && from_ref "$a" "$l"; then readlink -f $ALT/$l
    elif [ "$a" = r100 ] || from_ref "$a" "$l"; then readlink -f $RF/$l; else readlink -f $PK/$l; fi
  done
}
preload(){ want "$1" | tr '\n' ':' | sed 's/:$//'; }
arm_env(){ # arm_env ARM -> the environment assignments of the arm, one per line
  case $1 in
    pkg)  echo "LD_LIBRARY_PATH=$PK:$RT/lib";;
    r100) echo "LD_LIBRARY_PATH=$RF:/opt/rocm/core-10.0/lib";;
    *)    echo "LD_LIBRARY_PATH=$RT/lib"; echo "LD_PRELOAD=$(preload "$1")";;
  esac
}
arm_bin(){ if [ "$1" = r100 ]; then echo $RF/llama-batched-bench; else echo /opt/mx-llama.cpp/bin/llama-batched-bench; fi; }
# static check of every arm: the loader must map exactly the arm's libraries, and one ROCm tree
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; touch $DONE; exit 1; }
[ -s $TSV ] && { log "FATAL: $TSV exists; choose another TAG"; touch $DONE; exit 1; }
PC=$(cat /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT); RC=$(git -C /root/wt-master-v100 rev-parse HEAD)
ch=$(git -C /root/wt-master-v100 diff --name-only "$RC" "$PC" 2>/dev/null) || { log "FATAL: package commit $PC is not in the repository"; touch $DONE; exit 1; }
[ "$ch" = ggml/src/ggml-hip/CMakeLists.txt ] || { log "FATAL: package commit $PC differs from the reference in: $(echo $ch | cut -c1-200)"; touch $DONE; exit 1; }
log "package $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906), commit ${PC:0:9} = reference ${RC:0:9} + ggml/src/ggml-hip/CMakeLists.txt"
for a in $ARMS; do
  case $a in pkg|llama|llaman|ggml|hip|all|r100) ;; *) log "FATAL: unknown arm $a"; touch $DONE; exit 1;; esac
  mapfile -t E < <(arm_env $a)
  got=$(env "${E[@]}" ldd "$(arm_bin $a)" 2>/dev/null | awk '{ if ($2 == "=>") print $3; else print $1 }' | grep -E '/lib(ggml|llama)' | xargs -r -n1 readlink -f | sort -u)
  exp=$(want $a | sort -u)
  [ "$got" = "$exp" ] || { log "FATAL: arm $a maps other libraries than intended: $(echo "$got" | tr '\n' ' ')"; touch $DONE; exit 1; }
  rtw=/opt/rocm/core-10.1; [ "$a" = r100 ] && rtw=/opt/rocm/core-10.0
  nro=$(env "${E[@]}" ldd "$(arm_bin $a)" | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | wc -l)
  bad=$(env "${E[@]}" ldd "$(arm_bin $a)" | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | grep -vc "^$rtw/")
  [ "$bad" = 0 ] && [ "$nro" -gt 5 ] || { log "FATAL: arm $a resolves $bad of $nro ROCm libraries outside $rtw"; touch $DONE; exit 1; }
  log "arm $a: $(echo "$exp" | sed -e "s|$RF/|ref:|" -e "s|$PK/|pkg:|" -e "s|$ALT/|alt:|" -e 's/\.so[.0-9]*$//' | tr '\n' ' '); $nro ROCm libraries under $rtw"
done
log "n=$N, arms: $ARMS; load $(cut -d' ' -f1-3 /proc/loadavg)"

. $R/gpu-test-env.sh
export GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 GGML_TP_AR_MAX_NE=20481 HSA_FORCE_FINE_GRAIN_PCIE=1
BPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${BPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $W/$TAG-clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$W/$TAG-smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $W/$TAG-watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $W/$TAG-smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
log "$(state_line)"
cell(){ # cell ARM BLOCK
  local arm=$1 blk=$2 s=4 d=65536
  local out=$O/multi-$arm-$blk.md E child got exp
  mapfile -t E < <(arm_env $arm)
  env "${E[@]}" AMD_COMGR_CACHE_DIR=/root/.cache/comgr-swap-$arm timeout 7200 \
    "$(arm_bin $arm)" -m $M --device rocm0,rocm1,rocm2,rocm3 -sm tensor -ngl all \
    -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -c $(( s*(d+1280) )) -npp $d -ntg 1024 -npl $s \
    > $out 2>${out%.md}.log &
  BPID=$!
  sleep 45      # the libraries are mapped long before; the model is still loading
  child=$(pgrep -P $BPID | head -1)
  got=$(awk '$6 ~ /\/lib(ggml|llama)/ {print $6}' /proc/${child:-0}/maps 2>/dev/null | sort -u)
  exp=$(want $arm | sort -u)
  echo "$got" > ${out%.md}.maps
  wait $BPID
  local rc=$?
  BPID=""
  if [ "$got" != "$exp" ]; then
    log "WRONG LIBRARIES in $arm b$blk -- discarded: $(echo "$got" | tr '\n' ' ')"; printf "multi\t%s\t%s\tWRONGLIBS\tWRONGLIBS\n" "$arm" "$blk" >> $TSV; return 0
  fi
  if ! $NF "${out%.md}.log" >/dev/null 2>&1; then
    log "FA FALLBACK in $arm b$blk -- discarded"; printf "multi\t%s\t%s\tFALLBACK\tFALLBACK\n" "$arm" "$blk" >> $TSV; return 0
  fi
  if grep -qiE 'NCCL init failed|probe AllReduce failed' "${out%.md}.log"; then
    log "RCCL NOT UP in $arm b$blk -- discarded"; printf "multi\t%s\t%s\tNORCCL\tNORCCL\n" "$arm" "$blk" >> $TSV; return 0
  fi
  local m
  if ! m=$(python3 -I $CM "$out" $d $s 2>>$P); then
    log "CELL FAILED (rc=$rc): $arm b$blk"; printf "multi\t%s\t%s\tFAILED\tFAILED\n" "$arm" "$blk" >> $TSV; return 0
  fi
  printf "multi\t%s\t%s\t%s\n" "$arm" "$blk" "$m" >> $TSV
  log "$arm b$blk: $(echo "$m" | awk -F'\t' '{printf "decode %.4f, prefill %.1f",$1,$2}')"
}
for blk in $(seq 1 $N); do
  order=$(python3 -I -c 'import random,sys; a=sys.argv[2:]; random.Random(int(sys.argv[1])).shuffle(a); print(" ".join(a))' "$(( 20261116 + blk ))" $ARMS)
  log "block $blk order: $order"
  for a in $order; do cell $a $blk; done
done
log "decode per slot by arm: $(for a in $ARMS; do printf '%s [%s] ' "$a" "$(awk -F'\t' -v a="$a" '$2==a && $4+0>0 {printf "%s%.4f", (n++ ? " " : ""), $4}' $TSV)"; done)"
trap - INT TERM EXIT
kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"
restore
python3 -I - $TSV <<'PY' 2>&1 | tee -a $P
import collections, sys
d = collections.defaultdict(list)
for l in open(sys.argv[1]):
    r = l.rstrip('\n').split('\t')
    try: d[r[1]].append((float(r[3]), float(r[4])))
    except ValueError: print('discarded:', r)
base = 'all' if 'all' in d else None
for a in sorted(d):
    v = [x[0] for x in d[a]]; p = [x[1] for x in d[a]]
    line = '[resid] %-6s n=%d decode %.4f (%.4f-%.4f) prefill %.1f (%.1f-%.1f)' % (a, len(v), sum(v)/len(v), min(v), max(v), sum(p)/len(p), min(p), max(p))
    if base and a != base:
        b = [x[0] for x in d[base]]; bm = sum(b)/len(b)
        c = 'above' if min(v) > max(b) else 'below' if max(v) < min(b) else 'overlaps'
        line += '  vs all: %+.2f %%, %s' % (100*((sum(v)/len(v))/bm - 1), c)
    print(line)
if 'pkg' in d and 'all' in d:
    v = [x[0] for x in d['pkg']]; b = [x[0] for x in d['all']]
    eff = 100*((sum(v)/len(v))/(sum(b)/len(b)) - 1)
    print('[resid] the package has a share of its own: %s (effect %+.2f %%, every pkg sample below every "all" sample: %s)' % ('YES' if max(v) < min(b) and eff <= -0.10 else 'NO', eff, max(v) < min(b)))
PY
touch $DONE; log "=== PKG-RESID DONE ==="
