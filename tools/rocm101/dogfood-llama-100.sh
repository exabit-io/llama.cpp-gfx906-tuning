#!/bin/bash
# dogfood-llama-100.sh OUT VERSION -- after the ROCm 10.0-stream build is published into the ROLLING channel
# mx-llama-cpp-gfx906: install exactly VERSION from the channel the way the channel notes tell a ROCm 10.0 machine to
# (it is not the channel's newest version; that stays the ROCm 10.1 build), verify the chain InRelease signature ->
# Packages SHA256 -> deb SHA256 -> origin signature inside the deb, then install the channel's newest version again so
# the box ends on the ROCm 10.1 build it runs.
set -u
export LC_ALL=C
O=$1; VER=$2; mkdir -p $O
SRC=/etc/apt/sources.list.d/mx-llama-cpp-gfx906.sources
BASE=https://github.com/exabit-io/resize-amdgpu-bars/releases/download/mx-llama-cpp-gfx906
KR=/etc/apt/keyrings/exabit-apt.gpg
log(){ echo "$(date -Is) [dogfood] $*" | tee -a $O/progress; }
pgrep -x llama-batched-b >/dev/null && { log "FATAL: a timing run is in progress"; exit 1; }
grep -q "^URIs: $BASE\$" $SRC || { log "FATAL: $SRC is not the rolling channel"; exit 1; }
case $VER in *.rocm10.0.gfx906.*) ;; *) log "FATAL: $VER is not a rocm10.0 version"; exit 1;; esac
WAS=$(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906)
apt-get update > $O/update.out 2>&1; log "apt-get update rc=$?: $(grep -ciE '^(Err|W:|E:)' $O/update.out) error or warning lines"
NEWEST=$(apt-cache policy mx-llama-cpp-gfx906 | awk '/Candidate:/ {print $2}')
log "installed before: $WAS; candidate (newest): $NEWEST; versions in the channel: $(apt-cache madison mx-llama-cpp-gfx906 | awk -F'|' '{print $2}' | tr -s ' ' | tr '\n' ' ')"
case $NEWEST in *.rocm10.1.gfx906.*) ;; *) log "FATAL: the channel's newest version is not a ROCm 10.1 build"; exit 1;; esac
apt-get clean
DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1 apt-get install -y --allow-downgrades "mx-llama-cpp-gfx906=$VER" > $O/install.out 2>&1 < /dev/null
log "apt-get install mx-llama-cpp-gfx906=$VER rc=$?: $(grep -E '^Get:' $O/install.out | cut -c1-170)"
V=$(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906); log "installed: $V $(dpkg-query -W -f='${Status}' mx-llama-cpp-gfx906)"
[ "$V" = "$VER" ] || log "WRONG VERSION installed"
T=$(mktemp -d); cd $T || exit 1
curl -fsSL -o InRelease $BASE/InRelease; curl -fsSL -o Packages $BASE/Packages
gpg --no-default-keyring --keyring $KR --verify InRelease > gpg.out 2>&1; log "InRelease signature: $(grep -E 'Good signature|BAD' gpg.out | cut -c1-100)"
want=$(sha256sum Packages | cut -d' ' -f1); grep -q " $want .* Packages\$" InRelease && log "Packages SHA256 is in InRelease" || log "Packages SHA256 NOT in InRelease"
FN=$(awk -v v="$VER" '/^Package: /{p=$2} /^Version: /{ver=$2} /^Filename: /{f=$2} /^SHA256: /{s=$2} /^$/{if (p=="mx-llama-cpp-gfx906" && ver==v) print f, s}' Packages)
if [ -n "$FN" ]; then
  set -- $FN; curl -fsSL -o pkg.deb $BASE/$1; got=$(sha256sum pkg.deb | cut -d' ' -f1)
  [ "$got" = "$2" ] && log "deb SHA256 matches Packages ($1)" || log "deb SHA256 MISMATCH ($1)"
  mem=$(ar t pkg.deb | tr '\n' ' '); ar x pkg.deb
  cat debian-binary control.tar.* data.tar.* | gpg --no-default-keyring --keyring $KR --verify _gpgorigin - > origin.out 2>&1; log "origin signature in the deb (members: $mem): $(grep -E 'Good signature|BAD' origin.out | cut -c1-100)"
  log "local build == channel deb: $(cmp -s pkg.deb "$(cat /root/mx-llama-deb/last-deb.txt)" && echo identical || echo DIFFERENT)"
else
  log "version $VER is NOT in the channel's Packages"
fi
cd /; rm -rf $T
log "version: $(/opt/mx-llama.cpp/bin/llama-cli --version 2>&1 | grep version); source $(cat /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT)"
n=$(ldd /opt/mx-llama.cpp/lib/libggml-hip.so | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | wc -l)
out=$(ldd /opt/mx-llama.cpp/lib/libggml-hip.so | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | grep -vc '^/opt/rocm/core-10.0/')
log "ROCm libraries of libggml-hip.so: $n, outside /opt/rocm/core-10.0: $out; $(tr '\n' ';' < /usr/share/doc/mx-llama-cpp-gfx906/BUILD-INFO | cut -c1-260)"
# back to the channel's newest version, which is what this box runs
DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1 apt-get install -y mx-llama-cpp-gfx906 > $O/install-newest.out 2>&1 < /dev/null
log "apt-get install mx-llama-cpp-gfx906 (newest) rc=$?: $(grep -E '^Get:' $O/install-newest.out | cut -c1-170)"
V=$(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906)
out=$(ldd /opt/mx-llama.cpp/lib/libggml-hip.so | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | grep -vc '^/opt/rocm/core-10.1/')
log "box ends on: $V (was $WAS; newest $NEWEST); ROCm libraries outside /opt/rocm/core-10.1: $out"
[ "$V" = "$NEWEST" ] && [ "$out" = 0 ] && log "=== DOGFOOD DONE ===" || log "=== DOGFOOD: THE BOX IS NOT ON THE NEWEST VERSION ==="
