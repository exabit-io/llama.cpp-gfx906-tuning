#!/bin/bash
# publish-gfx906-rccl-topology.sh VERSION -- build gfx906-rccl-topology from a clean export of tag
# gfx906-rccl-topology-VERSION of exabit-io/llama.cpp-gfx906-tuning (tools/gfx906-rccl-topology/), sign the deb
# (origin signature), tag exabit-io/resize-amdgpu-bars with the same name, publish the per-version release (signed flat
# apt repo) and add the deb to the rolling channel (release gfx906-rccl-topology), which is what clients use.
#
# Before: commit and push the source with the version in debian/changelog and the script's VERSION, push the tag in
# the tuning repo, write $NOTES (per-version notes) and update $CHAN_NOTES (channel contents).
set -euo pipefail
export LC_ALL=C
V=${1:?usage: $0 VERSION}
TAG=gfx906-rccl-topology-$V
SRC=/root/llama.cpp-benchmarking
NOTES=/root/local-apt/gfx906-rccl-topology-$V-notes.md
CHAN_NOTES=/root/local-apt/gfx906-rccl-topology-channel-notes.md
OUT=/root/local-apt/gfx906-rccl-topology
CHAN=/root/local-apt/channel-gfx906-rccl-topology
export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519_exabit -o IdentitiesOnly=yes"
for f in "$NOTES" "$CHAN_NOTES"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done
grep -q '@@' "$NOTES" "$CHAN_NOTES" && { echo "unfilled placeholders in the notes" >&2; exit 1; }
mkdir -p "$OUT" "$CHAN"

echo "== $(date -Is) source: tag $TAG of the tuning repo"
cd "$SRC"
git fetch -q origin main "refs/tags/$TAG:refs/tags/$TAG"
commit=$(git rev-parse "$TAG^{commit}")
git merge-base --is-ancestor "$commit" origin/main || { echo "$TAG ($commit) is not on origin/main" >&2; exit 1; }

echo "== $(date -Is) build from a clean export of $commit"
B=$(mktemp -d "${TMPDIR:-/tmp}/rccltopo.XXXXXX"); trap 'rm -rf "$B"' EXIT
git archive "$commit" tools/gfx906-rccl-topology | tar -x -C "$B"
( cd "$B/tools/gfx906-rccl-topology" && dpkg-buildpackage -b -us -uc ) > "$OUT/build-$V.log" 2>&1 \
  || { tail -20 "$OUT/build-$V.log" >&2; exit 1; }
deb=$B/tools/gfx906-rccl-topology_${V}_all.deb
[ "$(dpkg-deb -f "$deb" Version)" = "$V" ] || { echo "built version is not $V" >&2; exit 1; }
if lintian -I --pedantic "$B"/tools/*.changes 2>&1 | grep -v 'root privileges' | grep -E '^[EW]:'; then
  echo "lintian errors or warnings" >&2; exit 1
fi
install -m 0644 "$deb" "$OUT/"
deb=$OUT/${deb##*/}

echo "== $(date -Is) sign deb"
/root/local-apt/sign-debs.sh "$deb"

echo "== $(date -Is) tag on exabit-io/resize-amdgpu-bars"
cd /root/resize-amdgpu-bars
git fetch -q origin
if ! git ls-remote --exit-code origin "refs/tags/$TAG" > /dev/null; then
  git tag -a "$TAG" "$(git rev-parse origin/main)" -m "gfx906-rccl-topology $V: exabit-io/llama.cpp-gfx906-tuning $commit (tag $TAG, tools/gfx906-rccl-topology/). Release assets are the signed flat apt repository; clients use the rolling channel gfx906-rccl-topology."
  git push origin "refs/tags/$TAG"
fi

echo "== $(date -Is) release $TAG"
LABEL=gfx906-rccl-topology DESC="RCCL topology for the XGMI ring of gfx906 GPUs" ARCHES="amd64 all" \
  /root/local-apt/github-release-pkg.sh "$TAG" "gfx906-rccl-topology $V (RCCL topology for the XGMI ring of gfx906 GPUs)" "$NOTES" "$deb"

echo "== $(date -Is) rolling channel gfx906-rccl-topology"
install -m 0644 "$deb" "$CHAN/"
REPO=exabit-io/resize-amdgpu-bars TAG=gfx906-rccl-topology CHAN="$CHAN" LABEL=gfx906-rccl-topology ARCHES="amd64 all" \
  TITLE="gfx906-rccl-topology: rolling apt channel (RCCL topology for gfx906 XGMI rings)" \
  DESC="RCCL topology for the XGMI ring of gfx906 GPUs" \
  /root/local-apt/github-channel.sh "$CHAN_NOTES"
echo "== $(date -Is) published"
