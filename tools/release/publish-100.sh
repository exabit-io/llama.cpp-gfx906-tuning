#!/bin/bash
# publish-100.sh DEB -- sign the mx-llama-cpp-gfx906 1.0.0 deb (origin signature), tag exabit-io/resize-amdgpu-bars,
# publish the per-version release (signed flat apt repo; notes and provenance), then add the deb to the ROLLING channel
# (release mx-llama-cpp-gfx906), which is what clients use: one source line, new versions arrive with apt upgrade.
# The channel starts with this release and also holds the two earlier versions (0.7.0, 0.9.0), byte-identical to their
# per-version releases.
set -euo pipefail
export LC_ALL=C
DEB=$1
V=1.0.0
NOTES=/root/mx-llama-deb/release-notes-$V.final.md
CHAN_NOTES=/root/local-apt/mx-llama-cpp-gfx906-channel-notes.md
TAG=mx-llama-cpp-$V-gfx906-rocm10.0
SRCTAG=gfx906/v$V/r0/master
CHAN=/root/local-apt/channel-mx-llama-cpp-gfx906
REL=/root/local-apt/github-release-pkg.sh
for f in "$DEB" "$NOTES" "$CHAN_NOTES"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done
grep -q '@@' "$NOTES" "$CHAN_NOTES" && { echo "unfilled placeholders in the notes" >&2; exit 1; }

echo "== $(date -Is) sign deb"
/root/local-apt/sign-debs.sh "$DEB"

echo "== $(date -Is) tag on exabit-io/resize-amdgpu-bars"
cd /root/resize-amdgpu-bars
export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519_exabit -o IdentitiesOnly=yes"
git fetch -q origin
VER=$(dpkg-deb -f "$DEB" Version)
SRC=$(dpkg-deb --fsys-tarfile "$DEB" | tar -xO ./usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT)
[ "$SRC" = "$(git -C /root/exabit-llama.cpp rev-parse "$SRCTAG^{commit}")" ] || { echo "deb source $SRC is not $SRCTAG" >&2; exit 1; }
if ! git ls-remote --exit-code origin "refs/tags/$TAG" > /dev/null; then
  git tag -a $TAG "$(git rev-parse origin/main)" -m "mx-llama-cpp-gfx906 $VER: exabit-io/mx-llama.cpp master $SRC (tag $SRCTAG), llama.cpp v$V, ROCm 10.0.0-gfx906+20261018172427. Release assets are the signed flat apt repository; clients use the rolling channel mx-llama-cpp-gfx906."
  git push origin refs/tags/$TAG
fi

echo "== $(date -Is) release $TAG"
LABEL=mx-llama-cpp-gfx906-rocm10 DESC="mx-llama.cpp for gfx906 on ROCm 10.0" \
  $REL $TAG "mx-llama-cpp-gfx906 $V (llama.cpp for gfx906 on ROCm 10.0)" "$NOTES" "$DEB"

echo "== $(date -Is) rolling channel mx-llama-cpp-gfx906"
mkdir -p "$CHAN"
for old in /root/mx-llama-deb/mx-llama-cpp-gfx906_0.7.0.git20261022.ef70ef51e.gfx906.20261018172427-1_amd64.deb \
           /root/mx-llama-deb/mx-llama-cpp-gfx906_0.9.0.git20261106.58db0123a.gfx906.20261018172427-1_amd64.deb; do
  [ -f "$CHAN/${old##*/}" ] || install -m 0644 "$old" "$CHAN/"
done
install -m 0644 "$DEB" "$CHAN/"
REPO=exabit-io/resize-amdgpu-bars TAG=mx-llama-cpp-gfx906 CHAN="$CHAN" LABEL=mx-llama-cpp-gfx906-rocm10 ARCHES="amd64" \
  TITLE="mx-llama-cpp-gfx906: rolling apt channel (llama.cpp for gfx906 on ROCm 10.0)" \
  DESC="mx-llama.cpp for gfx906 on ROCm 10.0" \
  /root/local-apt/github-channel.sh "$CHAN_NOTES"
echo "== $(date -Is) published"
