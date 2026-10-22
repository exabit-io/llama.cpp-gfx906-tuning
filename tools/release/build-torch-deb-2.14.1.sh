#!/bin/bash
# Build a Debian package of PyTorch (2.14.1) for gfx906 on ROCm 10.0.
# Wraps the wheels built by build-torch.sh (this directory) into a system package
# so the fleet can apt-install it (REQUIREMENTS R3.8: reproducible on every node).
set -u
SRC=/root/pytorch-gfx906-rocm10-2.14.1
V=$SRC/venv/lib/python3.12/site-packages
VER="$(cat $SRC/torch-version.txt)"
UPV=${VER%%+*}
ROCMSTAMP=${VER##*.}
ROCMVER="$(dpkg-query -W -f='${Version}' amdrocm10.0-gfx906)"
DEBVER="${VER}-1"
PKG=python3-torch-gfx906
OUT=/root/pytorch-deb-2.14.1
STAGE=$OUT/$PKG
log() { echo "$(date -Is) $*"; }

mkdir -p $OUT; rm -rf $STAGE; mkdir -p $STAGE/DEBIAN $STAGE/usr/lib/python3/dist-packages $STAGE/usr/share/doc/$PKG $STAGE/usr/bin

log "staging python packages ..."
for d in torch torchvision torchaudio torchgen functorch; do
  [ -d "$V/$d" ] && cp -a "$V/$d" $STAGE/usr/lib/python3/dist-packages/
done
for d in $V/torch-*.dist-info $V/torchvision-*.dist-info $V/torchaudio-*.dist-info; do
  [ -d "$d" ] && cp -a "$d" $STAGE/usr/lib/python3/dist-packages/
done
# console scripts shipped by the wheels
for b in torchrun torchfrtrace convert-caffe2-to-onnx convert-onnx-to-caffe2; do
  [ -f "$SRC/venv/bin/$b" ] && sed '1s|^#!.*|#!/usr/bin/python3|' "$SRC/venv/bin/$b" > $STAGE/usr/bin/$b && chmod 755 $STAGE/usr/bin/$b
done

# strip build-host paths that would be wrong on a target machine
find $STAGE/usr/lib/python3/dist-packages -name 'RECORD' -delete 2>/dev/null
find $STAGE/usr/lib/python3/dist-packages -name '*.pyc' -delete 2>/dev/null
find $STAGE/usr/lib/python3/dist-packages -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null

INSTALLED_KB=$(du -sk $STAGE | cut -f1)
log "staged ${INSTALLED_KB} KB"
TVV=$(ls -d $V/torchvision-*.dist-info | sed -E 's|.*/torchvision-([0-9.]+).*|\1|')
TAV=$(ls -d $V/torchaudio-*.dist-info | sed -E 's|.*/torchaudio-([0-9.]+).*|\1|')
MKLV=$($SRC/venv/bin/python -m pip show mkl-static 2>/dev/null | awk '/^Version/{print $2}')
HIPV=$(/opt/rocm/core-10.0/bin/hipconfig --version | cut -d- -f1)
log "torch $UPV torchvision $TVV torchaudio $TAV, ROCm $ROCMVER (HIP $HIPV)"

cat > $STAGE/DEBIAN/control <<EOF
Package: $PKG
Version: $DEBVER
Architecture: amd64
Maintainer: Exabit Networks <packages@exabit.io>
Installed-Size: $INSTALLED_KB
Depends: python3 (>= 3.12), python3 (<< 3.13), python3-filelock, python3-jinja2,
 python3-networkx, python3-sympy, python3-typing-extensions, python3-fsspec,
 python3-numpy, python3-setuptools, python3-pil, amdrocm10.0-gfx906 (>= $ROCMVER)
Section: science
Priority: optional
Homepage: https://github.com/exabit-io/resize-amdgpu-bars
Description: PyTorch $UPV for AMD gfx906 on ROCm 10.0
 Unmodified upstream PyTorch v$UPV, torchvision $TVV and torchaudio $TAV,
 built against ROCm $ROCMVER (HIP $HIPV) for the gfx906
 architecture (Radeon VII, Instinct MI50/MI60, Radeon Pro Vega II).
 .
 AMD dropped gfx906 from ROCm; this package fills the gap left by the upstream
 gfx906 project, which publishes wheels only up to ROCm 7.14.
 .
 Built with -march=native for Cascade Lake (AVX512F/DQ/BW/VL/CD + AVX512-VNNI),
 Intel oneMKL as the BLAS backend and oneDNN enabled. No PyTorch source patches
 are used; gfx906 support comes from PYTORCH_ROCM_ARCH.
 .
 NOTE: libtorch_cpu.so statically links Intel oneMKL, redistributed under the
 Intel Simplified Software License (see /usr/share/doc/$PKG/).
EOF

cat > $STAGE/DEBIAN/postinst <<'EOF'
#!/bin/sh
set -e
if [ "$1" = "configure" ]; then
    # byte-compile lazily; torch is large and py_compile on it is slow
    echo "python3-torch-gfx906: verifying import ..."
    if python3 -c 'import torch' >/dev/null 2>&1; then
        echo "python3-torch-gfx906: torch $(python3 -c 'import torch;print(torch.__version__)') OK"
    else
        echo "python3-torch-gfx906: WARNING: 'import torch' failed; check the ROCm 10.0 runtime" >&2
    fi
fi
exit 0
EOF
chmod 755 $STAGE/DEBIAN/postinst

cp /root/pytorch-gfx906-rocm10/licenses/Intel-oneMKL-LICENSE.txt /root/pytorch-gfx906-rocm10/licenses/README-third-party.md $STAGE/usr/share/doc/$PKG/ 2>/dev/null
cat > $STAGE/usr/share/doc/$PKG/copyright <<EOF
Upstream-Name: pytorch
Source: https://github.com/pytorch/pytorch

Files: *
Copyright: PyTorch contributors
License: BSD-3-Clause

Files: usr/lib/python3/dist-packages/torch/lib/libtorch_cpu.so
Comment: statically links Intel oneAPI Math Kernel Library $MKLV
License: Intel-Simplified-Software-License
 Full text in Intel-oneMKL-LICENSE.txt in this directory. Redistribution of the
 binaries without modification is expressly permitted.
EOF
printf '%s (%s) noble; urgency=medium\n\n  * PyTorch %s (torchvision %s, torchaudio %s) for gfx906 on ROCm %s, built by Exabit, Inc.\n\n -- Exabit Networks <packages@exabit.io>  %s\n' \
  "$PKG" "$DEBVER" "$UPV" "$TVV" "$TAV" "$ROCMVER" "$(date -R)" | gzip -9n > $STAGE/usr/share/doc/$PKG/changelog.Debian.gz

log "building deb ..."
dpkg-deb --root-owner-group -Zgzip -z6 --build $STAGE $OUT/${PKG}_${DEBVER//+/.}_amd64.deb
log "done: $(ls -la $OUT/*.deb | awk '{print $9, $5}')"
