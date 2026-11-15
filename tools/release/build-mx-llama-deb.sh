#!/bin/bash
# build-mx-llama-deb.sh SRC -- Debian package of exabit-io/mx-llama.cpp (master) for gfx906 on ROCm (stream ROCMSTREAM,
# default 10.1; ROCm streams install side by side under /opt/rocm/core-<stream>, so every ROCm path here names the stream).
#
# The tree's own defaults (GGML_HIP_RCCL=ON, GGML_CUDA_FA_QUANTS=all, GGML_TP_AR_MAX_NE=20481, max-ilp scheduler) plus
# -march=native on the host side (Cascade Lake, Mac Pro 7,1: AVX512F/DQ/BW/VL/CD + VNNI). Installs into /opt/mx-llama.cpp
# with DT_RPATH (not RUNPATH) on its own lib dir and then on the ROCm stream's lib dir, so no other llama.cpp/ggml install
# on the box (ld.so.conf.d entries, LD_LIBRARY_PATH) can shadow its libraries and the binaries load the ROCm they were
# built against whatever /opt/rocm's alternatives point at. HOST-CPU HEAVY: never beside a loaded four-die GPU job.
# Version: <upstream>+git<date>.<sha>.rocm<stream>.gfx906.<ROCm build stamp>-1 (the stream is named because two streams can
# carry the same stamp; it sorts above the earlier builds' <upstream>+git<date>.<sha>.gfx906.<stamp>).
# COMPILERSTREAM (default 10.0): the ROCm stream whose toolchain compiles and links the package; the package still depends
# on, and loads, ROCMSTREAM's runtime. Measured 2026-11-14 (rocm-ab.sh, rocm-cross.sh, swap screens): ROCm 10.1's LLVM 24
# generates a slower head-256 flash-attention tile kernel (prefill -1.4 % at 4x64K, -4.1 % at 1x255K, decode unchanged), and
# code from ROCm 10.0's LLVM 23 keeps its speed and passes the whole suite on the 10.1 runtime (same library sonames).
# Set COMPILERSTREAM=$ROCMSTREAM once a newer compiler passes the design-point A/B.
set -euo pipefail
SRC=${1:?usage: $0 SRC}
OUT=/root/mx-llama-deb
B=$OUT/build
PREFIX=/opt/mx-llama.cpp
PKG=mx-llama-cpp-gfx906
ROCMSTREAM=${ROCMSTREAM:-10.1}
COMPILERSTREAM=${COMPILERSTREAM:-10.0}
ROCM=/opt/rocm/core-$ROCMSTREAM            # the runtime the package depends on and loads
ROCMC=/opt/rocm/core-$COMPILERSTREAM       # the toolchain (compiler, headers, link libraries) it is built with
[ -d $ROCM/lib ] || { echo "no ROCm $ROCMSTREAM under $ROCM" >&2; exit 1; }
[ -x $ROCMC/lib/llvm/bin/clang++ ] || { echo "no ROCm $COMPILERSTREAM toolchain under $ROCMC" >&2; exit 1; }
log(){ echo "$(date -Is) $*" | tee -a $OUT/build.log; }

[ -z "$(git -C $SRC status --porcelain --untracked-files=no)" ] || { echo "dirty tree: $SRC" >&2; exit 1; }
SHA=$(git -C $SRC rev-parse HEAD); SHA9=${SHA:0:9}
DATE=$(git -C $SRC log -1 --format=%cd --date=format:%Y%m%d HEAD)
UPV=$(git -C $SRC describe --tags --match 'v[0-9]*' --abbrev=0 HEAD); UPV=${UPV#v}
BNUM=$(git -C $SRC describe --tags --match 'b[0-9]*' --abbrev=0 HEAD 2>/dev/null || echo b?)
ROCMSTAMP=$(dpkg-query -W -f='${Version}' amdrocm-core$ROCMSTREAM-gfx906 | sed -E 's/.*\+//')
VER="${UPV}+git${DATE}.${SHA9}.rocm${ROCMSTREAM}.gfx906.${ROCMSTAMP}"
DEBVER="${VER}-1"
CLANGV=$($ROCMC/lib/llvm/bin/clang++ --version | head -1)
log "=== $PKG $DEBVER from $SRC ($SHA, upstream v$UPV $BNUM), runtime ROCm $ROCMSTREAM build $ROCMSTAMP ($ROCM), compiled by ROCm $COMPILERSTREAM ($CLANGV) ==="

rm -rf $B; mkdir -p $B
ROCM_PATH=$ROCMC HIP_PATH=$ROCMC cmake -S $SRC -B $B -G "Unix Makefiles" -DCMAKE_PREFIX_PATH=$ROCMC \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_HIP_ARCHITECTURES=gfx906 -DCMAKE_HIP_COMPILER=$ROCMC/lib/llvm/bin/clang++ \
  -DGGML_NATIVE=ON -DCMAKE_C_FLAGS="-march=native" -DCMAKE_CXX_FLAGS="-march=native" \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF \
  "-DCMAKE_INSTALL_RPATH=$PREFIX/lib;$ROCM/lib" -DCMAKE_SKIP_BUILD_RPATH=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags" -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache \
  > $OUT/configure.log 2>&1
grep -E "GGML_HIP_RCCL|FA_QUANTS|MAX_ILP|march" $B/CMakeCache.txt | grep -v "^//" >> $OUT/build.log || true
t0=$(date +%s)
ROCM_PATH=$ROCMC HIP_PATH=$ROCMC nice -n 5 cmake --build $B -j 56 > $OUT/compile.log 2>&1
other=$(grep -E '^(hip|hipblas|rocblas|rccl)_DIR:' $B/CMakeCache.txt | grep -vc "=$ROCMC/" || true)
[ "$other" = 0 ] || { log "ROCm packages found outside $ROCMC: $(grep -E '^(hip|hipblas|rocblas|rccl)_DIR:' $B/CMakeCache.txt | tr '\n' ' ')"; exit 1; }
log "compiled in $(( $(date +%s) - t0 ))s, warnings $(grep -c ' warning: ' $OUT/compile.log), errors $(grep -c ' error: ' $OUT/compile.log)"

STAGE=$OUT/stage/$PKG
rm -rf $OUT/stage; mkdir -p $STAGE
DESTDIR=$STAGE cmake --install $B > $OUT/install.log 2>&1
# development files are not shipped: no headers, cmake/pkg-config files or static libs
rm -rf $STAGE$PREFIX/include $STAGE$PREFIX/lib/cmake $STAGE$PREFIX/lib/pkgconfig
find $STAGE$PREFIX -name '*.a' -delete
strip --strip-unneeded $STAGE$PREFIX/bin/* 2>/dev/null || true
find $STAGE$PREFIX/lib -name '*.so*' -type f -exec strip --strip-unneeded {} + 2>/dev/null || true

# every ELF must carry DT_RPATH on $PREFIX/lib (then the ROCm stream's lib dir) and resolve its own ggml/llama libraries there
bad=0
for f in $(find $STAGE$PREFIX -type f \( -perm -u+x -o -name '*.so*' \)); do
  file -b "$f" | grep -q ELF || continue
  rp=$(readelf -d "$f" | sed -n 's/.*(RPATH).*\[\(.*\)\]/\1/p')
  # first entry the package's own lib dir, every further entry the ROCm stream's (the HIP link adds that one a second time)
  case "$rp" in "$PREFIX/lib:$ROCM/lib"|"$PREFIX/lib:$ROCM/lib:$ROCM/lib") ;; *) echo "RPATH is not $PREFIX/lib:$ROCM/lib: $f ($rp)" >&2; bad=1;; esac
  readelf -d "$f" | grep -q '(RUNPATH)' && { echo "RUNPATH present: $f" >&2; bad=1; }
done
[ $bad = 0 ] || { log "RPATH check FAILED"; exit 1; }

# Depends from the ELF NEEDED entries: system and ROCm libraries are mapped to their packages
needed=$(find $STAGE$PREFIX -type f | while read -r f; do file -b "$f" | grep -q ELF && readelf -d "$f" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'; done | sort -u)
own=$(cd $STAGE$PREFIX/lib && ls)
deps=""
for so in $needed; do
  echo "$own" | grep -qx "$so" && continue
  path=$(ls $ROCM/lib/$so 2>/dev/null || true)          # the stream's own library first: /opt/rocm/lib follows the alternatives
  [ -n "$path" ] || path=$(ldconfig -p 2>/dev/null | awk -v s="$so" '$1==s && /x86-64/ && !f {print $NF; f=1}')    # no early exit: a closed pipe is a failure under pipefail
  [ -n "$path" ] || { log "unresolved NEEDED $so"; exit 1; }
  pkg=$(dpkg -S "$(readlink -f "$path")" 2>/dev/null | head -1 | cut -d: -f1) || true
  [ -n "$pkg" ] || pkg=$(dpkg -S "$path" 2>/dev/null | head -1 | cut -d: -f1) || true
  [ -n "$pkg" ] || { log "no package owns $path ($so)"; exit 1; }
  deps="$deps $pkg"
done
# a ROCm host library loads its gfx906 kernels (rocBLAS Tensile, RCCL) from the matching -gfx906 package at run time
for p in $deps; do
  case $p in amdrocm-*-host$ROCMSTREAM) g=${p%-host$ROCMSTREAM}$ROCMSTREAM-gfx906; dpkg-query -W "$g" >/dev/null 2>&1 && deps="$deps $g";; esac
done
deps=$(echo $deps | tr ' ' '\n' | sort -u)
DEPENDS=""
for p in $deps; do
  v=$(dpkg-query -W -f='${Version}' "$p")
  case $p in
    amdrocm*) DEPENDS="$DEPENDS, $p (>= $v)";;
    *) DEPENDS="$DEPENDS, $p";;
  esac
done
DEPENDS=${DEPENDS#, }
log "Depends: $DEPENDS"

mkdir -p $STAGE/DEBIAN $STAGE/usr/share/doc/$PKG
INSTALLED_KB=$(du -sk $STAGE | cut -f1)
cat > $STAGE/DEBIAN/control <<EOF
Package: $PKG
Version: $DEBVER
Architecture: amd64
Maintainer: Exabit Networks <packages@exabit.io>
Installed-Size: $INSTALLED_KB
Depends: $DEPENDS
Section: science
Priority: optional
Homepage: https://github.com/exabit-io/mx-llama.cpp
Description: llama.cpp for AMD gfx906 (mx-llama.cpp + Exabit patches) on ROCm $ROCMSTREAM
 exabit-io/mx-llama.cpp master $SHA9: mxxm-t/mx-llama.cpp (the gfx906 fork of
 llama.cpp) merged with llama.cpp v$UPV ($BNUM), plus Exabit's gfx906 patches
 (fused norm/residual/Q8_1 decode paths, folded gated-delta-net producers,
 16-column MMVQ, DPP warp reductions, all-reduce size gate, max-ilp scheduling).
 .
 Built for ROCm $ROCMSTREAM.0-gfx906+$ROCMSTAMP (compiled with ROCm $COMPILERSTREAM's
 toolchain) for gfx906 (Radeon VII, Instinct
 MI50/MI60, Radeon Pro Vega II) with the tree's defaults: RCCL on, every
 flash-attention KV quant pair compiled, -march=native for Cascade Lake
 (AVX512 + VNNI, Mac Pro 7,1). The CPU code needs AVX-512.
 .
 Installs into $PREFIX (bin/ and lib/, DT_RPATH on lib/ and on
 $ROCM/lib); nothing is placed on the default PATH.
 Run e.g. $PREFIX/bin/llama-server.
EOF
cat > $STAGE/usr/share/doc/$PKG/copyright <<EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: mx-llama.cpp
Source: https://github.com/exabit-io/mx-llama.cpp

Files: *
Copyright: The ggml authors, mxxm-t/mx-llama.cpp contributors, Exabit, Inc.
License: MIT
EOF
sed -n '1,200p' $SRC/LICENSE >> $STAGE/usr/share/doc/$PKG/copyright
printf '%s (%s) noble; urgency=medium\n\n  * exabit-io/mx-llama.cpp master %s (llama.cpp v%s %s), ROCm %s.0-gfx906+%s.\n\n -- Exabit Networks <packages@exabit.io>  %s\n' \
  "$PKG" "$DEBVER" "$SHA9" "$UPV" "$BNUM" "$ROCMSTREAM" "$ROCMSTAMP" "$(date -R)" | gzip -9n > $STAGE/usr/share/doc/$PKG/changelog.Debian.gz
echo "$SHA" > $STAGE/usr/share/doc/$PKG/SOURCE-COMMIT
printf 'source: exabit-io/mx-llama.cpp %s (llama.cpp v%s %s)\nruntime: ROCm %s.0-gfx906+%s (%s), loaded through DT_RPATH\ncompiler: ROCm %s toolchain, %s\nhost flags: -march=native (%s)\n' \
  "$SHA" "$UPV" "$BNUM" "$ROCMSTREAM" "$ROCMSTAMP" "$ROCM" "$COMPILERSTREAM" "$CLANGV" "$(gcc -march=native -Q --help=target 2>/dev/null | awk '/^ *-march=/{print $2; exit}')" > $STAGE/usr/share/doc/$PKG/BUILD-INFO

DEB=$OUT/${PKG}_${DEBVER//+/.}_amd64.deb
dpkg-deb --root-owner-group -Zgzip -z6 --build $STAGE $DEB > /dev/null
log "built $DEB ($(du -h $DEB | cut -f1)); lintian-free checks: $(dpkg-deb -I $DEB | grep -E '^ (Package|Version|Installed-Size)' | tr -s ' ' | tr '\n' ' ')"
echo $DEB > $OUT/last-deb.txt
