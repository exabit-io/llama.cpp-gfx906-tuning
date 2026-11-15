#!/bin/bash
# build-backport.sh DSC "BUILD_DEPS" [PROFILES] -- rebuild a newer Ubuntu source package for Ubuntu 24.04 in the clean
# noble chroot (/var/cache/exabit-backports/noble-buildd, made with mmdebstrap --variant=buildd).
#   DSC         a .dsc in /root/backports-noble/src (fetched with apt-get source from the newer series: signature and
#               hashes checked by apt against the Ubuntu archive key)
#   BUILD_DEPS  "auto" = apt-get build-dep with the profiles; or an explicit package list, then dpkg-buildpackage -d
#               (for sources whose documentation build-deps are not profile-guarded: no docs are built)
#   PROFILES    default "nocheck nodoc"
# The only change to the source is a changelog entry: version <newer>~24.04.exabit1, distribution noble.
# Results: /root/backports-noble/out/<source>/ (debs, buildinfo, changes, build log).
set -euo pipefail
export LC_ALL=C
W=/root/backports-noble; C=/var/cache/exabit-backports/noble-buildd
DSC=$1; BDEPS=$2; PROFILES=${3:-nocheck nodoc}
[ -f "$W/src/$DSC" ] || { echo "no $W/src/$DSC" >&2; exit 1; }
SRCNAME=$(awk '/^Source:/{print $2; exit}' "$W/src/$DSC"); VER=$(awk '/^Version:/{print $2; exit}' "$W/src/$DSC")
NEWVER="$VER~24.04.exabit1"
B=$W/build/$SRCNAME; OUT=$W/out/$SRCNAME
rm -rf "$B" "$OUT"; mkdir -p "$B" "$OUT"
for f in $(awk '/^Files:/{f=1; next} /^[A-Za-z-]+:/{f=0} f && NF==3 && $1 ~ /^[0-9a-f]{32}$/ {print $3}' "$W/src/$DSC") "$DSC"; do cp -p "$W/src/$f" "$B/"; done
cat > "$B/run.sh" <<EOS
#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive LC_ALL=C DEBFULLNAME="Exabit Networks" DEBEMAIL=packages@exabit.io
cd /build
apt-get update -qq
dpkg-source -x $DSC src > /dev/null
cd src
if [ "$BDEPS" = auto ]; then
  apt-get build-dep -y -qq --build-profiles=$(echo $PROFILES | tr ' ' ',') ./ > /build/build-dep.log 2>&1
  DFLAG=""
else
  apt-get install -y -qq --no-install-recommends $BDEPS > /build/build-dep.log 2>&1
  DFLAG="-d"
fi
dch --force-bad-version --newversion "$NEWVER" --distribution noble --force-distribution \
  "Backport of $SRCNAME $VER (Ubuntu 26.04) to Ubuntu 24.04 by Exabit, Inc.: no source changes; built with profiles '$PROFILES'."
DEB_BUILD_OPTIONS="$PROFILES" DEB_BUILD_PROFILES="$PROFILES" dpkg-buildpackage -b -uc -us \$DFLAG > /build/build.log 2>&1
EOS
chmod 755 "$B/run.sh"
systemd-nspawn -q -D "$C" --bind="$B:/build" --resolv-conf=copy-host --register=no --pipe /build/run.sh
rc=$?
cp -p "$B"/*.deb "$B"/*.buildinfo "$B"/*.changes "$B"/build.log "$B"/build-dep.log "$OUT/" 2>/dev/null || true
echo "$SRCNAME $NEWVER rc=$rc: $(ls "$OUT"/*.deb 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
exit $rc
