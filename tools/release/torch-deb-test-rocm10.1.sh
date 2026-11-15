#!/bin/bash
# deb-test.sh OUT -- the python3-torch-gfx906 deb named in last-deb.txt (the ROCm 10.1 build), before publishing; the
# reference is the package installed now, i.e. the same PyTorch built against ROCm 10.0:
#  1. reference on the package installed now (the previous release): stack_check.py and ops_check.py, system python
#  2. install the new deb from the local file
#  3. properties of the installed files: unresolved libraries, which ROCm packages the libraries come from and whether
#     the deb's Depends reaches them, the Python requirements the wheel declares, the build flags torch reports
#  4. the same two checks on the new package
# Run on an idle host (the checks time GPU matmuls and an RCCL all-reduce; a build beside them skews both).
set -u
R=/root/pytorch-gfx906-rocm10.1-2.15.0; O=$1; mkdir -p $O; P=$O/progress
DEB=$(cat $R/last-deb.txt)
DEPDEBS=$(tr '\n' ' ' < $R/dep-debs.txt)              # the backports the deb depends on (typing_extensions, sympy)
PKG=python3-torch-gfx906
SP=/usr/lib/python3/dist-packages
log(){ echo "$(date -Is) [torch-deb] $*" | tee -a $P; }
[ -f "$DEB" ] || { log "missing deb: $DEB"; exit 1; }
for f in $DEPDEBS; do [ -f "$f" ] || { log "missing dependency deb: $f"; exit 1; }; done
. /root/rocm-tests/bench/gpu-test-env.sh
trap 'restore' EXIT
setmax
run_checks(){ # run_checks TAG -- both checks with the system interpreter, isolated mode, from /
  local t=$1 rc
  (cd / && timeout 1800 /usr/bin/python3 -I $R/stack_check.py) > $O/$t-stack.out 2> $O/$t-stack.err; rc=$?
  log "$t stack_check rc=$rc: $(grep -E '^torch |matmul 4096|RCCL all_reduce|ALL OK' $O/$t-stack.out | sed 's/^ *//' | tr '\n' ';' | cut -c1-900)"
  (cd / && timeout 1800 /usr/bin/python3 -I $R/ops_check.py) > $O/$t-ops.out 2> $O/$t-ops.err; rc=$?
  log "$t ops_check rc=$rc: $(grep -E '^ops_check:' $O/$t-ops.out); $(grep -E '^(FAIL|skip) ' $O/$t-ops.out | cut -c1-260 | tr '\n' ';')"
  log "$t distributed: $(grep -E '^(ok|FAIL|skip) +distributed' $O/$t-ops.out | sed 's/^ok *//' | cut -c1-200 | tr '\n' ';')"
  (cd / && timeout 1800 /usr/bin/python3 -I $R/pydeps_check.py) > $O/$t-pydeps.out 2> $O/$t-pydeps.err; rc=$?
  log "$t pydeps_check rc=$rc: $(grep -E '^torch ' $O/$t-pydeps.out | cut -c1-160); $(grep -E '^pydeps_check:' $O/$t-pydeps.out); $(grep -E '^FAIL ' $O/$t-pydeps.out | cut -c1-200 | tr '\n' ';')"
}
OLD=$(dpkg-query -W -f='${Version}' $PKG 2>/dev/null); NEW=$(dpkg-deb -f $DEB Version)
log "deb $DEB ($NEW, $(stat -c %s $DEB) bytes); installed now: ${OLD:-none}; load $(cut -d' ' -f1-3 /proc/loadavg)"
if [ "$OLD" != "$NEW" ]; then
  [ -n "$OLD" ] && run_checks ref
  DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1 apt-get install -y --allow-downgrades $DEPDEBS "$DEB" > $O/install.out 2>&1 < /dev/null
  log "apt-get install rc=$? -> $(dpkg-query -W -f='${Package} ${Version} ${Status}; ' $PKG python3-typing-extensions python3-sympy ninja-build 2>&1 | tr '\n' ' '); postinst: $(grep -E 'python3-torch-gfx906: ' $O/install.out | tail -1)"
fi
# libraries: nothing unresolved; every ROCm library's package must be reachable from the deb's Depends
: > $O/needed.txt
for f in $SP/torch/lib/*.so* $SP/torch/_C*.so $SP/torchvision/*.so $SP/torchaudio/lib/*.so; do
  [ -f "$f" ] || continue
  LD_LIBRARY_PATH= ldd "$f" 2>/dev/null | awk -v f="${f##*/}" '{print f, $1, $3}' >> $O/needed.txt
done
log "ELF files checked: $(cut -d' ' -f1 $O/needed.txt | sort -u | wc -l); unresolved libraries: $(grep -c 'not$\|not found' $O/needed.txt) $(grep 'not$\|not found' $O/needed.txt | awk '{print $2}' | sort -u | tr '\n' ' ' | cut -c1-200)"
awk '$3 ~ /^\/opt\/rocm/ {print $3}' $O/needed.txt | sort -u | xargs -r -n1 readlink -f | sort -u > $O/rocm-libs.txt
xargs -r dpkg -S < $O/rocm-libs.txt 2>/dev/null | cut -d: -f1 | sort -u > $O/rocm-pkgs.txt
META=$(dpkg-deb -f $DEB Depends | tr ',' '\n' | grep -o 'amdrocm[^ ]*' | head -1)
apt-cache depends --recurse --no-recommends --no-suggests --no-conflicts --no-breaks --no-replaces --no-enhances "$META" 2>/dev/null | grep -oE '^[a-z0-9][^ ]*|Depends: [^ <]+' | sed 's/Depends: //' | sort -u > $O/meta-closure.txt
log "ROCm libraries linked: $(wc -l < $O/rocm-libs.txt) from packages: $(tr '\n' ' ' < $O/rocm-pkgs.txt | cut -c1-600)"
log "ROCm packages NOT reachable from $META: $(comm -23 $O/rocm-pkgs.txt $O/meta-closure.txt | tr '\n' ' ') (empty = the Depends line covers them)"
# Python requirements the wheel declares, against what the system interpreter has
(cd / && /usr/bin/python3 -I - > $O/pydeps.txt 2>&1 <<'EOF'
import importlib.metadata as md, re
from packaging.requirements import Requirement
for dist in ("torch", "torchvision", "torchaudio"):
    for r in md.requires(dist) or []:
        q = Requirement(r)
        if q.marker is not None and not q.marker.evaluate({"extra": ""}):
            continue
        try:
            v = md.version(q.name)
            ok = q.specifier.contains(v, prereleases=True)
        except md.PackageNotFoundError:
            v, ok = "MISSING", False
        print(f"{dist}: {q.name}{q.specifier} -> {v} {'ok' if ok else 'NOT SATISFIED'}")
EOF
)
log "python requirements: $(grep -c ' ok$' $O/pydeps.txt) satisfied; not satisfied: $(grep -v ' ok$' $O/pydeps.txt | tr '\n' ';' | cut -c1-400)"
(cd / && /usr/bin/python3 -I -c 'import torch; print(torch.__config__.show())') > $O/config.txt 2>&1
log "build flags: $(grep -oE -- '-march=[a-z0-9-]+|BLAS_INFO=[a-z]+|USE_ROCM=[A-Z]+|USE_MKLDNN=[A-Z]+|USE_NCCL=[A-Z0-9]+|ROCM_VERSION[^,]*|LAPACK_INFO=[a-z]+|CXX_COMPILER=[^,]+' $O/config.txt | sort -u | tr '\n' ' ' | cut -c1-400)"
run_checks new
log "=== TORCH DEB TEST DONE ==="
