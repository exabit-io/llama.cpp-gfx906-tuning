#!/bin/bash
# publish.sh LLAMA_DEB TORCH_DEB -- sign both debs (origin signature), tag exabit-io/resize-amdgpu-bars, and publish one
# signed flat apt repo per package as a GitHub release (github-release-pkg.sh signs Release/InRelease).
set -euo pipefail
LDEB=$1; TDEB=$2
NOTES_L=/root/mx-llama-deb/release-notes.final.md
NOTES_T=/root/pytorch-gfx906-rocm10-2.14.1/release-notes.final.md
TAG_L=mx-llama-cpp-0.7.0-gfx906-rocm10.0
TAG_T=pytorch-2.14.1-gfx906-rocm10.0
REL=/root/local-apt/github-release-pkg.sh
for f in "$LDEB" "$TDEB" "$NOTES_L" "$NOTES_T"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done
grep -q '@@' "$NOTES_L" "$NOTES_T" && { echo "unfilled placeholders in notes" >&2; exit 1; }

echo "== $(date -Is) sign debs"
/root/local-apt/sign-debs.sh "$LDEB" "$TDEB"

echo "== $(date -Is) tags on exabit-io/resize-amdgpu-bars"
cd /root/resize-amdgpu-bars
export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519_exabit -o IdentitiesOnly=yes"
git fetch -q origin
base=$(git rev-parse origin/main)
LSHA=$(dpkg-deb -f "$LDEB" Version); TVER=$(dpkg-deb -f "$TDEB" Version)
git tag -f -a $TAG_L $base -m "mx-llama-cpp-gfx906 $LSHA: exabit-io/mx-llama.cpp master $(cat /root/mx-llama-deb/stage/mx-llama-cpp-gfx906/usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT), ROCm 10.0.0-gfx906+20261018172427. Release assets are the signed flat apt repository."
git tag -f -a $TAG_T $base -m "python3-torch-gfx906 $TVER: pytorch v2.14.1 (5c48869), built by /root/pytorch-gfx906-rocm10-2.14.1/build-torch.sh against ROCm 10.0.0-gfx906+20261018172427. Release assets are the signed flat apt repository."
git push origin refs/tags/$TAG_L refs/tags/$TAG_T

echo "== $(date -Is) release $TAG_L"
LABEL=mx-llama-cpp-gfx906-rocm10 DESC="mx-llama.cpp for gfx906 on ROCm 10.0" \
  $REL $TAG_L "mx-llama-cpp-gfx906 0.7.0 (llama.cpp for gfx906 on ROCm 10.0)" "$NOTES_L" "$LDEB"

echo "== $(date -Is) release $TAG_T"
LABEL=pytorch-gfx906-rocm10 DESC="PyTorch 2.14.1 for gfx906 on ROCm 10.0" \
  $REL $TAG_T "python3-torch-gfx906 2.14.1 (PyTorch for gfx906 on ROCm 10.0)" "$NOTES_T" "$TDEB" \
  /root/pytorch-gfx906-rocm10/licenses/Intel-oneMKL-LICENSE.txt /root/pytorch-gfx906-rocm10/licenses/README-third-party.md
echo "== $(date -Is) published"
