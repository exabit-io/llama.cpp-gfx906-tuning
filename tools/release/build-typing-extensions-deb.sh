#!/bin/bash
# build-typing-extensions-deb.sh [VERSION] -- python3-typing-extensions backport for Ubuntu 24.04 (noble ships 4.10.0).
# PyTorch 2.15 imports typing_extensions.Format and get_annotations (4.13.0) from torch.jit at `import torch`, so
# python3-torch-gfx906 2.15 cannot be imported with noble's package. This repackages the upstream wheel from PyPI
# (one pure-Python file) with the same file layout, package name and maintainer scripts as Ubuntu's package, so apt
# upgrades it in place and python3-torch-gfx906 can depend on python3-typing-extensions (>= 4.13.0).
# Version default: the one in the build venv, i.e. the one PyTorch was built and tested with.
set -euo pipefail
export LC_ALL=C
R=/root/pytorch-gfx906-rocm10-2.15.0
PY=$R/venv/bin/python
V=${1:-$($PY -I -c 'import importlib.metadata as m; print(m.version("typing_extensions"))')}
PKG=python3-typing-extensions
DEBVER="$V-0exabit1"                     # sorts above noble's 4.10.0-1 and below any distro upload of the same upstream version
OUT=/root/pytorch-deb-2.15.0
W=$R/typing-extensions-$V                # the download and its extraction, nothing else
STAGE=$OUT/$PKG
DP=$STAGE/usr/lib/python3/dist-packages
log() { echo "$(date -Is) $*"; }

rm -rf "$W" "$STAGE"; mkdir -p "$W/wheel" "$W/x" "$DP" "$STAGE/DEBIAN" "$STAGE/usr/share/doc/$PKG"
$PY -I -m pip download -q --no-deps --only-binary=:all: --dest "$W/wheel" "typing-extensions==$V"
WHL=$(ls "$W"/wheel/typing_extensions-"$V"-py3-none-any.whl)
SHA=$(sha256sum "$WHL" | cut -d' ' -f1)
# the same digest from PyPI's JSON API (a second read of what PyPI published for this file)
PYPI=$(curl -fsSL "https://pypi.org/pypi/typing-extensions/$V/json" | $PY -I -c 'import json,sys; d=json.load(sys.stdin); print(next(u["digests"]["sha256"] for u in d["urls"] if u["filename"].endswith("py3-none-any.whl")))')
[ "$SHA" = "$PYPI" ] || { echo "wheel sha256 $SHA differs from PyPI's $PYPI" >&2; exit 1; }
log "wheel ${WHL##*/} sha256 $SHA (matches PyPI)"
$PY -I -m zipfile -e "$WHL" "$W/x"
[ "$(ls "$W/x" | tr '\n' ' ')" = "typing_extensions-$V.dist-info typing_extensions.py " ] || { echo "unexpected wheel contents: $(ls "$W/x" | tr '\n' ' ')" >&2; exit 1; }
cmp "$W/x/typing_extensions.py" "$R/venv/lib/python3.12/site-packages/typing_extensions.py" \
  || { echo "the wheel's module differs from the one in the build venv" >&2; exit 1; }

install -m 0644 "$W/x/typing_extensions.py" "$DP/typing_extensions.py"
DI=$DP/typing_extensions-$V.dist-info; mkdir -p "$DI"
install -m 0644 "$W/x/typing_extensions-$V.dist-info/METADATA" "$W/x/typing_extensions-$V.dist-info/WHEEL" "$DI/"
echo debian > "$DI/INSTALLER"
LICENSE=$(find "$W/x/typing_extensions-$V.dist-info" -name 'LICENSE*' | head -1)
[ -f "$LICENSE" ] || { echo "no LICENSE in the wheel" >&2; exit 1; }
{
  cat <<EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: typing-extensions
Source: https://pypi.org/project/typing-extensions/$V/
Comment: Repackaged from the upstream wheel ${WHL##*/}
 (sha256 $SHA) by Exabit, Inc. for Ubuntu 24.04,
 which ships 4.10.0. No source changes.

Files: *
Copyright: Python Software Foundation and the typing_extensions contributors
License: PSF-2.0

License: PSF-2.0
EOF
  sed -e 's/^$/./' -e 's/^/ /' "$LICENSE"
} > "$STAGE/usr/share/doc/$PKG/copyright"
printf '%s (%s) noble; urgency=medium\n\n  * Backport of typing_extensions %s for Ubuntu 24.04, which ships 4.10.0:\n    the upstream wheel, repackaged without changes.\n  * Needed by python3-torch-gfx906 2.15, which imports\n    typing_extensions.Format and get_annotations (4.13.0) at import time.\n\n -- Exabit Networks <packages@exabit.io>  %s\n' \
  "python-typing-extensions" "$DEBVER" "$V" "$(date -R)" | gzip -9n > "$STAGE/usr/share/doc/$PKG/changelog.Debian.gz"

cat > "$STAGE/DEBIAN/control" <<EOF
Package: $PKG
Source: python-typing-extensions
Version: $DEBVER
Architecture: all
Maintainer: Exabit Networks <packages@exabit.io>
Installed-Size: $(du -sk "$STAGE/usr" | cut -f1)
Depends: python3:any
Section: python
Priority: optional
Homepage: https://typing-extensions.readthedocs.io/en/latest/
Description: Backported and Experimental Type Hints for Python (backport for Ubuntu 24.04)
 The typing_extensions module: new type system features for older Python
 versions, and experimentation with new typing PEPs before they are accepted.
 .
 This is typing_extensions $V, the unmodified upstream wheel, packaged by
 Exabit, Inc. to replace the 4.10.0 that Ubuntu 24.04 ships. PyTorch 2.15
 (python3-torch-gfx906) needs 4.13.0 or newer to be imported.
EOF
# the maintainer scripts dh_python3 generates for this package (byte-compile on install, clean on removal)
cat > "$STAGE/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
if command -v py3compile >/dev/null 2>&1; then
	py3compile -p python3-typing-extensions
fi
exit 0
EOF
cat > "$STAGE/DEBIAN/prerm" <<'EOF'
#!/bin/sh
set -e
if command -v py3clean >/dev/null 2>&1; then
	py3clean -p python3-typing-extensions
else
	dpkg -L python3-typing-extensions | sed -En -e '/^(.*)\/(.+)\.py$/s,,rm "\1/__pycache__/\2".*,e'
	find /usr/lib/python3/dist-packages/ -type d -name __pycache__ -empty -print0 | xargs --null --no-run-if-empty rmdir
fi
exit 0
EOF
chmod 755 "$STAGE/DEBIAN/postinst" "$STAGE/DEBIAN/prerm"
(cd "$STAGE" && find usr -type f -exec md5sum {} + | sort -k2 > DEBIAN/md5sums)
DEB=$OUT/${PKG}_${DEBVER}_all.deb
dpkg-deb --root-owner-group -Zgzip -z9 --build "$STAGE" "$DEB" > /dev/null
echo "$DEB" > $R/last-typing-extensions-deb.txt
log "done: $DEB $(stat -c %s "$DEB") bytes; $(dpkg-deb -f "$DEB" Package Version Architecture | tr '\n' ' ')"
dpkg-deb -c "$DEB" | awk '{print $1, $3, $6}' | grep -v '/$'
