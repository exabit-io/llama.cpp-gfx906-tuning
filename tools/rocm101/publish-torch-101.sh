#!/bin/bash
# publish-torch-101.sh DEB -- the ROCm 10.1 build of python3-torch-gfx906 2.15.0 with the backports it depends on
# (python3-typing-extensions, already published; python3-sympy 1.14.0 with isympy-common and isympy3, rebuilt from
# Ubuntu 26.04's source): sign, tag, per-version release pytorch-<v>-gfx906-rocm10.1 (every deb it needs, so it resolves
# by itself), then the ROLLING channel python3-torch-gfx906, where this build becomes the newest version.
set -euo pipefail
export LC_ALL=C
DEB=$1
V=2.15.0; STREAM=10.1
R=/root/pytorch-gfx906-rocm$STREAM-$V
NOTES=$R/release-notes.final.md
CHAN_NOTES=/root/local-apt/python3-torch-gfx906-channel-notes.md
TAG=pytorch-$V-gfx906-rocm$STREAM
CHAN=/root/local-apt/channel-python3-torch-gfx906
REL=/root/local-apt/github-release-pkg.sh
LIC=/root/pytorch-gfx906-rocm10/licenses
REPO=exabit-io/resize-amdgpu-bars
TEDEB=$CHAN/python3-typing-extensions_4.16.0-0exabit1_all.deb       # the published, signed file
SYDEBS=$(ls /root/backports-noble/out/sympy/python3-sympy_*.deb /root/backports-noble/out/sympy/isympy-common_*.deb /root/backports-noble/out/sympy/isympy3_*.deb)
for f in "$DEB" "$TEDEB" $SYDEBS "$NOTES" "$CHAN_NOTES" $LIC/Intel-oneMKL-LICENSE.txt $LIC/README-third-party.md; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done
grep -q '@@' "$NOTES" "$CHAN_NOTES" && { echo "unfilled placeholders in the notes" >&2; exit 1; }
VER=$(dpkg-deb -f "$DEB" Version)
STAMP=${VER##*.gfx906.}; STAMP=${STAMP%-*}     # the ROCm build stamp in the version
case $VER in "$V+rocm$STREAM.gfx906."*) ;; *) echo "deb version $VER is not $V on ROCm $STREAM" >&2; exit 1;; esac
dpkg-deb -f "$DEB" Depends | grep -q "amdrocm$STREAM-gfx906 " || { echo "deb does not depend on amdrocm$STREAM-gfx906" >&2; exit 1; }
for old in $(ls $CHAN/python3-torch-gfx906_*.deb); do dpkg --compare-versions "$VER" gt "$(dpkg-deb -f "$old" Version)" || { echo "$VER does not sort above $(basename "$old")" >&2; exit 1; }; done
SRC=$(git -C $R/src/pytorch rev-parse HEAD)
[ "$(git -C $R/src/pytorch describe --tags --exact-match 2>/dev/null)" = "v$V" ] || { echo "source checkout is not v$V" >&2; exit 1; }

echo "== $(date -Is) sign debs"
/root/local-apt/sign-debs.sh "$DEB" $SYDEBS "$TEDEB"

echo "== $(date -Is) tag on $REPO"
cd /root/resize-amdgpu-bars
export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519_exabit -o IdentitiesOnly=yes"
git fetch -q origin
if ! git ls-remote --exit-code origin "refs/tags/$TAG" > /dev/null; then
  git tag -a $TAG "$(git rev-parse origin/main)" -m "python3-torch-gfx906 $VER: pytorch v$V (${SRC:0:7}), built by build-torch.sh against ROCm $STREAM.0-gfx906+$STAMP. Release assets are the signed flat apt repository; clients use the rolling channel python3-torch-gfx906."
  git push origin refs/tags/$TAG
fi

echo "== $(date -Is) release $TAG"
LABEL=pytorch-gfx906-rocm10 DESC="PyTorch $V for gfx906 on ROCm $STREAM" ARCHES="amd64 all" \
  $REL $TAG "python3-torch-gfx906 $V on ROCm $STREAM (PyTorch for gfx906)" "$NOTES" "$DEB" "$TEDEB" $SYDEBS \
  $LIC/Intel-oneMKL-LICENSE.txt $LIC/README-third-party.md

echo "== $(date -Is) rolling channel python3-torch-gfx906"
install -m 0644 "$DEB" $SYDEBS "$CHAN/"
REPO=$REPO TAG=python3-torch-gfx906 CHAN="$CHAN" LABEL=pytorch-gfx906-rocm10 ARCHES="amd64 all" \
  TITLE="python3-torch-gfx906: rolling apt channel (PyTorch for gfx906 on ROCm 10)" \
  DESC="PyTorch for gfx906 on ROCm 10" \
  /root/local-apt/github-channel.sh "$CHAN_NOTES" $LIC/Intel-oneMKL-LICENSE.txt $LIC/README-third-party.md
echo "== $(date -Is) published"
