#!/bin/bash
# dogfood-torch.sh OUT -- after publish-torch.sh: point this box at the ROLLING channel python3-torch-gfx906, install the
# package from it, verify the chain (InRelease signature -> Packages SHA256 -> deb SHA256 -> origin signature inside the
# deb) and run the stack check with the system interpreter.
set -u
export LC_ALL=C
R=/root/pytorch-gfx906-rocm10-2.15.0
O=$1; mkdir -p $O
PKG=python3-torch-gfx906
SRC=/etc/apt/sources.list.d/pytorch-gfx906.sources
BASE=https://github.com/exabit-io/resize-amdgpu-bars/releases/download/python3-torch-gfx906
KR=/etc/apt/keyrings/exabit-apt.gpg
log(){ echo "$(date -Is) [dogfood] $*" | tee -a $O/progress; }
[ -f /var/backups/pytorch-gfx906.sources.pre-channel ] || cp -a $SRC /var/backups/pytorch-gfx906.sources.pre-channel
cat > $SRC <<EOF
# PyTorch for gfx906 on ROCm 10.0 (python3-torch-gfx906), built and signed by Exabit, Inc.: the ROLLING channel (GitHub
# release python3-torch-gfx906 on exabit-io/resize-amdgpu-bars), a signed flat apt repository holding every version.
# Since 2026-11-14; replaced the per-version line (backup /var/backups/pytorch-gfx906.sources.pre-channel).
Types: deb
URIs: $BASE
Suites: ./
Signed-By: $KR
EOF
apt-get update > $O/update.out 2>&1; log "apt-get update rc=$?: $(grep -ciE '^(Err|W:|E:)' $O/update.out) error or warning lines; channel: $(grep -c 'releases/download/python3-torch-gfx906 ' $O/update.out) index lines"
log "candidates: $(apt-cache policy $PKG | sed -n '2,3p' | tr -s ' ' | tr '\n' ';'); versions in the channel: $(apt-cache madison $PKG | awk -F'|' '{print $2}' | tr -s ' ' | tr '\n' ' ')"
apt-get clean
DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1 apt-get install -y --reinstall $PKG python3-typing-extensions > $O/install.out 2>&1 < /dev/null; log "apt-get install --reinstall rc=$?: $(grep -E '^Get:' $O/install.out | cut -c1-150 | tr '\n' ';')"
log "installed: $(dpkg-query -W -f='${Package} ${Version} ${Status}; ' $PKG python3-typing-extensions)"
T=$(mktemp -d); cd $T
curl -fsSL -o InRelease $BASE/InRelease; curl -fsSL -o Packages $BASE/Packages
gpg --no-default-keyring --keyring $KR --verify InRelease > gpg.out 2>&1; log "InRelease signature: $(grep -E 'Good signature|BAD' gpg.out | cut -c1-100)"
want=$(sha256sum Packages | cut -d' ' -f1); grep -q " $want .* Packages\$" InRelease && log "Packages SHA256 is in InRelease" || log "Packages SHA256 NOT in InRelease"
for pair in "$PKG:$(cat $R/last-deb.txt)" "python3-typing-extensions:$(cat $R/last-typing-extensions-deb.txt)"; do
  p=${pair%%:*}; localdeb=${pair#*:}
  V=$(dpkg-query -W -f='${Version}' $p)
  FN=$(awk -v p="$p" -v v="$V" '/^Package: /{pk=$2} /^Version: /{ver=$2} /^Filename: /{f=$2} /^SHA256: /{s=$2} /^$/{if (pk==p && ver==v) print f, s}' Packages)
  [ -n "$FN" ] || { log "$p: installed version $V is NOT in the channel's Packages"; exit 1; }
  set -- $FN; rm -rf x; mkdir x; curl -fsSL -o x/pkg.deb $BASE/$1; got=$(sha256sum x/pkg.deb | cut -d' ' -f1)
  [ "$got" = "$2" ] && log "$p: deb SHA256 matches Packages ($1)" || log "$p: deb SHA256 MISMATCH ($1)"
  mem=$(ar t x/pkg.deb | tr '\n' ' '); (cd x && ar x pkg.deb && cat debian-binary control.tar.* data.tar.* | gpg --no-default-keyring --keyring $KR --verify _gpgorigin - > origin.out 2>&1)
  log "$p: origin signature in the deb (members: $mem): $(grep -E 'Good signature|BAD' x/origin.out | cut -c1-100); local build == channel deb: $(cmp -s x/pkg.deb "$localdeb" && echo identical || echo DIFFERENT)"
done
cd /; rm -rf $T
. /root/rocm-tests/bench/gpu-test-env.sh
trap 'restore' EXIT
setmax
(cd / && timeout 1800 /usr/bin/python3 -I $R/stack_check.py) > $O/stack.out 2> $O/stack.err; rc=$?
log "stack_check on the channel install rc=$rc: $(grep -E '^torch |RCCL all_reduce|ALL OK' $O/stack.out | sed 's/^ *//' | tr '\n' ';' | cut -c1-400)"
log "=== DOGFOOD DONE ==="
