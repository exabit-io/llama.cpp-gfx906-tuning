#!/bin/bash
# tile2-build.sh -- second round on clang 24's slower head-256 tile kernel: build the variants and print their static
# properties; the timing screen (screen.sh, 1 x 131072) is started separately with the arms chosen from this output.
#   c23none   clang 23 with the compiler's default scheduler (no -amdgpu-sched-strategy): does clang 23 need max-ilp to
#             be fast, i.e. is the difference the max-ilp schedule or something every schedule shares?
#   x23on23, x24on24   optimizer and code generator of the same compiler through the cross method (method controls)
#   x23on24   clang 23's optimized bitcode, clang 24's code generator      x24on23   the reverse (if clang 23 reads it)
#   vNN       clang 24 as built plus -mllvm -amdgpu-vgpr-threshold-percent=NN (new in LLVM 24: scales the register
#             limits the scheduler works to), NN = 95 90 85 80 75 70 60 50
# Compiles only (about a minute of a few cores): not during a timing run.
set -u
W=/root/rocm101-migration-20261114; C=$W/codegen; P=$W/tile2-build.progress
log(){ echo "$(date -Is) [tile2] $*" | tee -a $P; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
mkdir -p /root/swap; cd $C || exit 1
BP=(); NAMES=()
bg(){ # bg NAME TOOL ARGS...
  local n=$1 t=$2; shift 2
  python3 -I $C/$t "$n" "$@" > $C/drv-$n.out 2>&1 &
  BP+=($!); NAMES+=("$n")
}
bg c23none drvexp.py 100 --drop amdgpu-sched-strategy
bg x23on23 ircross.py 100 100
bg x24on24 ircross.py 101 101
bg x23on24 ircross.py 100 101
bg x24on23 ircross.py 101 100
for n in 95 90 85 80 75 70 60 50; do bg v$n drvexp.py 101 -mllvm -amdgpu-vgpr-threshold-percent=$n; done
for p in "${BP[@]}"; do wait "$p"; done
E=()
for a in "${NAMES[@]}"; do
  log "$(tr '\n' ' ' < $C/drv-$a.out | cut -c1-420)"
  [ -f $C/drv-$a.elf ] && [ -f /root/swap/$a/libggml-hip.so.1.0.0 ] && E+=("$C/drv-$a.elf")
done
python3 -I $C/loopstat.py $C/drv-c23.elf $C/drv-c24.elf $C/drv-c24old.elf "${E[@]}" 2>&1 | tee -a $P
python3 -I - $C/drv-c23.elf $C/drv-c24.elf $C/drv-c24old.elf "${E[@]}" <<'EOF' 2>&1 | tee -a $P
import difflib, os, sys
sys.path.insert(0, os.path.dirname(sys.argv[1]))
from asmcmp import insns, loops, sym
sub = 'flash_attn_tileILi256ELi256ELi16ELi2ELb0'
def hot(e):
    ins = insns(e, sym(e, sub)); s, t = max(loops(ins), key=lambda x: x[1] - x[0])
    return [m + ' ' + o for _, m, o in ins[s:t + 1]]
ref = {os.path.basename(e): hot(e) for e in sys.argv[1:3]}
print('hot loop of the 16x2 kernel, instruction text compared with the two controls (1.000 = identical):')
for e in sys.argv[1:]:
    try: h = hot(e)
    except SystemExit as x: print('  %-22s %s' % (os.path.basename(e), x)); continue
    print('  %-22s %s' % (os.path.basename(e), '   '.join('%s %.3f%s' % (k, difflib.SequenceMatcher(None, h, v, autojunk=False).ratio(), ' (same)' if h == v else '') for k, v in ref.items())))
EOF
log "=== TILE2 BUILDS DONE ==="
