#!/bin/bash
# dogfood-llama-101.sh OUT -- after publish-llama-101.sh: update from the ROLLING channel mx-llama-cpp-gfx906, reinstall the package
# from it, and verify the chain: InRelease signature -> Packages SHA256 -> deb SHA256 -> origin signature inside the deb.
set -u
export LC_ALL=C
mkdir -p "$1"; O=$(readlink -f "$1")   # absolute: the script changes directory below (a relative OUT lost the log lines after it, 2026-11-16)
SRC=/etc/apt/sources.list.d/mx-llama-cpp-gfx906.sources
BASE=https://github.com/exabit-io/resize-amdgpu-bars/releases/download/mx-llama-cpp-gfx906
KR=/etc/apt/keyrings/exabit-apt.gpg
log(){ echo "$(date -Is) [dogfood] $*" | tee -a $O/progress; }
[ -f /var/backups/mx-llama-cpp-gfx906.sources.pre-channel ] || cp -a $SRC /var/backups/mx-llama-cpp-gfx906.sources.pre-channel
cat > $SRC <<EOF
# mx-llama.cpp (exabit-io/mx-llama.cpp master) for gfx906 on ROCm 10, built and signed by Exabit, Inc.: the ROLLING
# channel (GitHub release mx-llama-cpp-gfx906 on exabit-io/resize-amdgpu-bars), a signed flat apt repository holding
# every version (ROCm 10.0 and 10.1 builds). Since 2026-11-14; replaced the per-version line (backup
# /var/backups/mx-llama-cpp-gfx906.sources.pre-channel).
Types: deb
URIs: $BASE
Suites: ./
Signed-By: $KR
EOF
apt-get update > $O/update.out 2>&1; log "apt-get update rc=$?: $(grep -ciE '^(Err|W:|E:)' $O/update.out) error or warning lines; channel: $(grep -c 'releases/download/mx-llama-cpp-gfx906 ' $O/update.out) index lines"
log "candidates: $(apt-cache policy mx-llama-cpp-gfx906 | sed -n '2,3p' | tr -s ' ' | tr '\n' ';'); versions in the channel: $(apt-cache madison mx-llama-cpp-gfx906 | awk -F'|' '{print $2}' | tr -s ' ' | tr '\n' ' ')"
apt-get clean
DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1 apt-get install -y --reinstall mx-llama-cpp-gfx906 > $O/install.out 2>&1 < /dev/null; log "apt-get install --reinstall rc=$?: $(grep -E '^Get:' $O/install.out | cut -c1-160)"
log "installed: $(dpkg-query -W -f='${Version} ${Status}' mx-llama-cpp-gfx906)"
T=$(mktemp -d); cd $T
curl -fsSL -o InRelease $BASE/InRelease; curl -fsSL -o Packages $BASE/Packages
gpg --no-default-keyring --keyring $KR --verify InRelease > gpg.out 2>&1; log "InRelease signature: $(grep -E 'Good signature|BAD' gpg.out | cut -c1-100)"
want=$(sha256sum Packages | cut -d' ' -f1); if grep -q " $want .* Packages\$" InRelease; then log "Packages SHA256 is in InRelease"; else log "Packages SHA256 NOT in InRelease"; fi
V=$(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906)
FN=$(awk -v v="$V" '/^Package: /{p=$2} /^Version: /{ver=$2} /^Filename: /{f=$2} /^SHA256: /{s=$2} /^$/{if (p=="mx-llama-cpp-gfx906" && ver==v) print f, s}' Packages)
[ -n "$FN" ] || { log "installed version $V is NOT in the channel's Packages"; exit 1; }
set -- $FN; curl -fsSL -o pkg.deb $BASE/$1; got=$(sha256sum pkg.deb | cut -d' ' -f1)
if [ "$got" = "$2" ]; then log "deb SHA256 matches Packages ($1)"; else log "deb SHA256 MISMATCH ($1)"; fi
mem=$(ar t pkg.deb | tr '\n' ' '); ar x pkg.deb
cat debian-binary control.tar.* data.tar.* | gpg --no-default-keyring --keyring $KR --verify _gpgorigin - > origin.out 2>&1; log "origin signature in the deb (members: $mem): $(grep -E 'Good signature|BAD' origin.out | cut -c1-100)"
log "local build == channel deb: $(cmp -s pkg.deb "$(cat /root/mx-llama-deb/last-deb.txt)" && echo identical || echo DIFFERENT)"
cd /; rm -rf $T
log "version: $(/opt/mx-llama.cpp/bin/llama-cli --version 2>&1 | grep version); source $(cat /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT)"
case $V in *.rocm10.1.gfx906.*) log "installed version is the ROCm 10.1 build";; *) log "installed version $V is NOT the ROCm 10.1 build";; esac
n=$(ldd /opt/mx-llama.cpp/lib/libggml-hip.so | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | wc -l)
out=$(ldd /opt/mx-llama.cpp/lib/libggml-hip.so | awk '$3 ~ /^\/opt\/rocm/ {print $3}' | xargs -r -n1 readlink -f | grep -vc '^/opt/rocm/core-10.1/')
log "ROCm libraries of libggml-hip.so: $n, outside /opt/rocm/core-10.1: $out; $(tr '\n' ';' < /usr/share/doc/mx-llama-cpp-gfx906/BUILD-INFO | cut -c1-260)"
log "=== DOGFOOD DONE ==="
