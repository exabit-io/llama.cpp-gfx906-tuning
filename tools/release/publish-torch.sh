#!/bin/bash
# publish-torch.sh DEB -- sign the python3-torch-gfx906 2.15.0 deb (origin signature), tag exabit-io/resize-amdgpu-bars,
# publish the per-version release (signed flat apt repo; notes and provenance), then add the deb to the ROLLING channel
# (release python3-torch-gfx906), which is what clients use: one source line, new versions arrive with apt upgrade.
# The channel starts with this release and also holds the two earlier versions (2.13.0, 2.14.1), byte-identical to the
# assets of their per-version releases (checked here against the digests GitHub reports).
# With the PyTorch deb goes the python3-typing-extensions backport it depends on (noble's 4.10.0 is too old for 2.15):
# both debs are in the per-version release and in the channel, so either repository satisfies the dependency by itself.
set -euo pipefail
export LC_ALL=C
DEB=$1
TEDEB=$(cat /root/pytorch-gfx906-rocm10-2.15.0/last-typing-extensions-deb.txt)
V=2.15.0
R=/root/pytorch-gfx906-rocm10-$V
NOTES=$R/release-notes.final.md
CHAN_NOTES=/root/local-apt/python3-torch-gfx906-channel-notes.md
TAG=pytorch-$V-gfx906-rocm10.0
CHAN=/root/local-apt/channel-python3-torch-gfx906
REL=/root/local-apt/github-release-pkg.sh
LIC=/root/pytorch-gfx906-rocm10/licenses
REPO=exabit-io/resize-amdgpu-bars
for f in "$DEB" "$TEDEB" "$NOTES" "$CHAN_NOTES" $LIC/Intel-oneMKL-LICENSE.txt $LIC/README-third-party.md; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done
grep -q '@@' "$NOTES" "$CHAN_NOTES" && { echo "unfilled placeholders in the notes" >&2; exit 1; }
VER=$(dpkg-deb -f "$DEB" Version)
[ "${VER%%+*}" = "$V" ] || { echo "deb version $VER is not $V" >&2; exit 1; }
SRC=$(git -C $R/src/pytorch rev-parse HEAD)
STAMP=${VER##*.gfx906.}; STAMP=${STAMP%-*}     # the ROCm build stamp in the version
[ "$(git -C $R/src/pytorch describe --tags --exact-match 2>/dev/null)" = "v$V" ] || { echo "source checkout is not v$V" >&2; exit 1; }

echo "== $(date -Is) channel contents: the earlier versions must be the published bytes"
mkdir -p "$CHAN"
old214=/root/pytorch-deb-2.14.1/python3-torch-gfx906_2.14.1.gfx906.20261018172427-1_amd64.deb
[ -f "$CHAN/${old214##*/}" ] || install -m 0644 "$old214" "$CHAN/"
for pair in pytorch-2.13.0-gfx906-rocm10.0:python3-torch-gfx906_2.13.0.gfx906.20260917140126-1_amd64.deb \
            pytorch-2.14.1-gfx906-rocm10.0:python3-torch-gfx906_2.14.1.gfx906.20261018172427-1_amd64.deb; do
  t=${pair%%:*}; n=${pair#*:}
  [ -f "$CHAN/$n" ] || { echo "missing $CHAN/$n (download it from release $t)" >&2; exit 1; }
  want=$(gh api "repos/$REPO/releases/tags/$t" -q ".assets[] | select(.name==\"$n\") | .digest")
  got="sha256:$(sha256sum "$CHAN/$n" | cut -d' ' -f1)"
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "$n differs from the asset of $t ($got vs $want)" >&2; exit 1; }
  echo "  $n = asset of $t"
done

echo "== $(date -Is) sign debs"
/root/local-apt/sign-debs.sh "$DEB" "$TEDEB"

echo "== $(date -Is) tag on $REPO"
cd /root/resize-amdgpu-bars
export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519_exabit -o IdentitiesOnly=yes"
git fetch -q origin
if ! git ls-remote --exit-code origin "refs/tags/$TAG" > /dev/null; then
  git tag -a $TAG "$(git rev-parse origin/main)" -m "python3-torch-gfx906 $VER: pytorch v$V (${SRC:0:7}), built by build-torch.sh against ROCm 10.0.0-gfx906+$STAMP. Release assets are the signed flat apt repository; clients use the rolling channel python3-torch-gfx906."
  git push origin refs/tags/$TAG
fi

echo "== $(date -Is) release $TAG"
LABEL=pytorch-gfx906-rocm10 DESC="PyTorch $V for gfx906 on ROCm 10.0" ARCHES="amd64 all" \
  $REL $TAG "python3-torch-gfx906 $V (PyTorch for gfx906 on ROCm 10.0)" "$NOTES" "$DEB" "$TEDEB" \
  $LIC/Intel-oneMKL-LICENSE.txt $LIC/README-third-party.md

echo "== $(date -Is) rolling channel python3-torch-gfx906"
install -m 0644 "$DEB" "$TEDEB" "$CHAN/"
REPO=$REPO TAG=python3-torch-gfx906 CHAN="$CHAN" LABEL=pytorch-gfx906-rocm10 ARCHES="amd64 all" \
  TITLE="python3-torch-gfx906: rolling apt channel (PyTorch for gfx906 on ROCm 10.0)" \
  DESC="PyTorch for gfx906 on ROCm 10.0" \
  /root/local-apt/github-channel.sh "$CHAN_NOTES" $LIC/Intel-oneMKL-LICENSE.txt $LIC/README-third-party.md
echo "== $(date -Is) published"
