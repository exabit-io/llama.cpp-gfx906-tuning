#!/bin/bash
# publish-090.sh DEB -- sign the mx-llama-cpp-gfx906 0.9.0 deb (origin signature), tag exabit-io/resize-amdgpu-bars, and
# publish it as a signed flat apt repo in a GitHub release (github-release-pkg.sh signs Release/InRelease).
set -euo pipefail
DEB=$1
NOTES=/root/mx-llama-deb/release-notes-0.9.0.final.md
TAG=mx-llama-cpp-0.9.0-gfx906-rocm10.0
REL=/root/local-apt/github-release-pkg.sh
for f in "$DEB" "$NOTES"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done
grep -q '@@' "$NOTES" && { echo "unfilled placeholders in notes" >&2; exit 1; }

echo "== $(date -Is) sign deb"
/root/local-apt/sign-debs.sh "$DEB"

echo "== $(date -Is) tag on exabit-io/resize-amdgpu-bars"
cd /root/resize-amdgpu-bars
export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519_exabit -o IdentitiesOnly=yes"
git fetch -q origin
base=$(git rev-parse origin/main)
VER=$(dpkg-deb -f "$DEB" Version)
SRC=$(cat /root/mx-llama-deb/stage/mx-llama-cpp-gfx906/usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT)
git tag -a $TAG $base -m "mx-llama-cpp-gfx906 $VER: exabit-io/mx-llama.cpp master $SRC (tag gfx906/v0.9.0/r0/master), llama.cpp v0.9.0, ROCm 10.0.0-gfx906+20261018172427. Release assets are the signed flat apt repository."
git push origin refs/tags/$TAG

echo "== $(date -Is) release $TAG"
LABEL=mx-llama-cpp-gfx906-rocm10 DESC="mx-llama.cpp for gfx906 on ROCm 10.0" \
  $REL $TAG "mx-llama-cpp-gfx906 0.9.0 (llama.cpp for gfx906 on ROCm 10.0)" "$NOTES" "$DEB"
echo "== $(date -Is) published"
