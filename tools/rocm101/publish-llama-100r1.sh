#!/bin/bash
# publish-llama-100r1.sh DEB -- the ROCm 10.0-stream build of mx-llama-cpp-gfx906 1.0.0 again: master 290fc642a (tag
# gfx906/v1.0.0/r1/master), compiled by ROCm 10.0's compiler, no global -march=native. It replaces, for machines that
# stay on ROCm 10.0, the build of 2026-11-14 (df2c6e5c5) that carried the flag. Sign, tag exabit-io/resize-amdgpu-bars,
# per-version release mx-llama-cpp-1.0.0-gfx906-rocm10.0-r1, then add the deb to the ROLLING channel
# mx-llama-cpp-gfx906. The channel keeps every earlier build, and its NEWEST version must stay the ROCm 10.1 build
# (this version sorts below it: ...290fc642a.rocm10.0... < ...290fc642a.rocm10.1...), so `apt upgrade` does not move
# a ROCm 10.1 machine to it.
set -euo pipefail
export LC_ALL=C
DEB=$1
V=1.0.0; STREAM=10.0
NOTES=/root/mx-llama-deb/release-notes-$V-rocm$STREAM-r1.final.md
CHAN_NOTES=/root/local-apt/mx-llama-cpp-gfx906-channel-notes.md
TAG=mx-llama-cpp-$V-gfx906-rocm$STREAM-r1
SRCTAG=gfx906/v$V/r1/master
CHAN=/root/local-apt/channel-mx-llama-cpp-gfx906
REL=/root/local-apt/github-release-pkg.sh
for f in "$DEB" "$NOTES" "$CHAN_NOTES"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done
grep -q '@@' "$NOTES" "$CHAN_NOTES" && { echo "unfilled placeholders in the notes" >&2; exit 1; }
VER=$(dpkg-deb -f "$DEB" Version)
STAMP=${VER##*.gfx906.}; STAMP=${STAMP%-*}     # the ROCm build stamp in the version
case $VER in *".rocm$STREAM.gfx906."*) ;; *) echo "deb version $VER is not a ROCm $STREAM build" >&2; exit 1;; esac
dpkg-deb -f "$DEB" Depends | grep -q "amdrocm-runtime$STREAM " || { echo "deb does not depend on amdrocm-runtime$STREAM" >&2; exit 1; }
grep -qF "$VER" "$CHAN_NOTES" || { echo "the channel notes do not name $VER" >&2; exit 1; }
above=0
for old in $(ls $CHAN/*.deb); do
  o=$(dpkg-deb -f "$old" Version)
  [ "$o" = "$VER" ] && [ "$(readlink -f "$old")" != "$(readlink -f "$DEB")" ] && ! cmp -s "$old" "$DEB" && { echo "$VER is already in the channel with other content" >&2; exit 1; }
  if dpkg --compare-versions "$o" gt "$VER"; then
    case $o in *.rocm10.1.gfx906.*) above=$((above + 1));; *) echo "$VER sorts below the non-10.1 build $o" >&2; exit 1;; esac
  else
    case $o in *.rocm10.1.gfx906.*) echo "note: $VER sorts above the ROCm 10.1 build $o (an older one)";; esac
  fi
done
[ "$above" -ge 1 ] || { echo "$VER would become the channel's newest version: refused" >&2; exit 1; }

echo "== $(date -Is) sign deb"
/root/local-apt/sign-debs.sh "$DEB"

echo "== $(date -Is) tag on exabit-io/resize-amdgpu-bars"
cd /root/resize-amdgpu-bars
export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519_exabit -o IdentitiesOnly=yes"
git fetch -q origin
SRC=$(dpkg-deb --fsys-tarfile "$DEB" | tar -xO ./usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT)
[ "$SRC" = "$(git -C /root/exabit-llama.cpp rev-parse "$SRCTAG^{commit}")" ] || { echo "deb source $SRC is not $SRCTAG" >&2; exit 1; }
[ "$(git -C /root/exabit-llama.cpp ls-remote exabit-mx "refs/tags/$SRCTAG^{}" | cut -f1)" = "$SRC" ] || { echo "$SRCTAG is not on exabit-io/mx-llama.cpp (or is another commit)" >&2; exit 1; }
if ! git ls-remote --exit-code origin "refs/tags/$TAG" > /dev/null; then
  git tag -a $TAG "$(git rev-parse origin/main)" -m "mx-llama-cpp-gfx906 $VER: exabit-io/mx-llama.cpp master $SRC (tag $SRCTAG), llama.cpp v$V, ROCm $STREAM.0-gfx906+$STAMP. Release assets are the signed flat apt repository; clients use the rolling channel mx-llama-cpp-gfx906."
  git push origin refs/tags/$TAG
fi

echo "== $(date -Is) release $TAG"
LABEL=mx-llama-cpp-gfx906-rocm10 DESC="mx-llama.cpp for gfx906 on ROCm $STREAM" \
  $REL $TAG "mx-llama-cpp-gfx906 $V on ROCm $STREAM, rebuilt without the global -march=native (llama.cpp for gfx906)" "$NOTES" "$DEB"

echo "== $(date -Is) rolling channel mx-llama-cpp-gfx906"
install -m 0644 "$DEB" "$CHAN/"
REPO=exabit-io/resize-amdgpu-bars TAG=mx-llama-cpp-gfx906 CHAN="$CHAN" LABEL=mx-llama-cpp-gfx906-rocm10 ARCHES="amd64" \
  TITLE="mx-llama-cpp-gfx906: rolling apt channel (llama.cpp for gfx906 on ROCm 10)" \
  DESC="mx-llama.cpp for gfx906 on ROCm 10" \
  /root/local-apt/github-channel.sh "$CHAN_NOTES"
echo "== $(date -Is) published"
