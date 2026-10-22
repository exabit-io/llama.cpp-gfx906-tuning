#!/bin/bash
# Build PyTorch 2.14.1 + vision + audio from source against ROCm 10.0 (gfx906 build 20261018172427) for gfx906.
#
# Port of /root/build-torch-rocm10.sh (2.13.0, ROCm build 20260917140126), itself a bare-metal port of
# mixa3607/ML-gfx906 pytorch/build-whl.Dockerfile (their 2.14.1 recipe, e9da9c3, is the same stock build).
# Stock pytorch v2.14.1 -- gfx906 comes from PYTORCH_ROCM_ARCH. -march=native (Cascade Lake, AVX512 + VNNI),
# static oneMKL, oneDNN on. HOST-CPU HEAVY: never beside a loaded four-die GPU job.
set -u
ROOT=/root/pytorch-gfx906-rocm10-2.14.1
LOG=$ROOT/logs
SUFFIX="gfx906.$(dpkg-query -W -f='${Version}' amdrocm-core10.0-gfx906 | sed -E 's/.*\+//')"
log() { echo "$(date -Is) $*" | tee -a $LOG/build.log; }
mkdir -p $ROOT/src $LOG $ROOT/dist

export ROCM_PATH=/opt/rocm/core-10.0
export HIP_PATH=$ROCM_PATH
export PATH=$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:/usr/local/lib
export PYTORCH_ROCM_ARCH=gfx906
export ROCM_ARCH=gfx906
export USE_ROCM=ON USE_CUDA=OFF
export USE_FLASH_ATTENTION=OFF USE_MEM_EFF_ATTENTION=OFF
export USE_NATIVE_ARCH=ON
export CFLAGS="-march=native -O3" CXXFLAGS="-march=native -O3"
export USE_MKLDNN=ON
export BLAS=MKL USE_STATIC_MKL=ON
export MAX_JOBS=${MAX_JOBS:-48}
export BUILD_TEST=0

log "=== torch build start: ROCM_PATH=$ROCM_PATH ($($ROCM_PATH/bin/hipconfig --version)) arch=$PYTORCH_ROCM_ARCH suffix=$SUFFIX jobs=$MAX_JOBS ==="
log "toolchain: $($ROCM_PATH/llvm/bin/clang --version | head -1)"

if [ ! -x $ROOT/venv/bin/python ]; then
  python3 -m venv $ROOT/venv || { log "FATAL: venv creation failed"; exit 1; }
fi
export PATH=$ROOT/venv/bin:$PATH
PY=$ROOT/venv/bin/python
$PY -m pip install -q --upgrade pip setuptools setuptools_scm wheel packaging build "scikit-build-core>=1.0" pyyaml six typing-extensions numpy cmake ninja jinja2 mkl-static mkl-include \
  > $LOG/pip-base.log 2>&1 || { log "FATAL: base pip install failed"; tail -5 $LOG/pip-base.log; exit 1; }
log "venv ready: $($PY --version), cmake $($ROOT/venv/bin/cmake --version | head -1 | awk '{print $3}'), ninja $($ROOT/venv/bin/ninja --version), mkl $($PY -m pip show mkl-static 2>/dev/null | awk '/^Version/{print $2}')"

[ -d $ROOT/src/pytorch/.git ] || { log "FATAL: no pytorch checkout in $ROOT/src/pytorch"; exit 1; }
cd $ROOT/src/pytorch
log "pytorch at $(git describe --tags 2>/dev/null || git rev-parse --short HEAD) ($(git rev-parse HEAD))"
$PY -m pip install -q -r requirements.txt > $LOG/pip-reqs.log 2>&1 \
  || { log "FATAL: requirements.txt install failed"; tail -8 $LOG/pip-reqs.log; exit 1; }

log "running tools/amd_build/build_amd.py (hipify) ..."
$PY tools/amd_build/build_amd.py > $LOG/hipify.log 2>&1 || { log "FATAL: hipify failed"; tail -8 $LOG/hipify.log; exit 1; }

VERSION="$(sed -E 's/(a|b|rc)[0-9]+$//' version.txt)"
export PYTORCH_BUILD_VERSION="${VERSION}+${SUFFIX}" PYTORCH_BUILD_NUMBER=0
export CMAKE_PREFIX_PATH=$($PY -c 'import sys; print(sys.prefix)')
echo "$PYTORCH_BUILD_VERSION" > $ROOT/torch-version.txt
log "building torch $PYTORCH_BUILD_VERSION ..."
t0=$(date +%s)
# 2.14 builds through scikit-build-core (PEP 517); setup.py bdist_wheel is gone.
# With -march=native, PyTorch's FindAVX accepts its first (empty) AVX512 flag and caches CXX_AVX512_FLAGS=' '; only fbgemm
# reads those flags, so its AVX512 sources lose -mavx512bf16 and v2.14's BF16 intrinsics do not compile. Seed the value a
# non-native build computes on this host (upstream wheels compile those files the same way and dispatch at run time).
AVX512_FLAGS="-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512bf16 -mfma"
CMAKE_ARGS="-DCXX_AVX512_FOUND=TRUE -DC_AVX512_FOUND=TRUE '-DCXX_AVX512_FLAGS=$AVX512_FLAGS' '-DC_AVX512_FLAGS=$AVX512_FLAGS'" \
$PY -m build --wheel --no-isolation --outdir $ROOT/dist > $LOG/build-torch.log 2>&1 \
  || { log "TORCH BUILD FAILED after $(( $(date +%s)-t0 ))s: $(grep -m3 -iE 'error:' $LOG/build-torch.log | cut -c1-160 | tr '\n' ' ')"; exit 1; }
log "torch wheel built in $(( $(date +%s)-t0 ))s: $(ls $ROOT/dist/torch-*.whl | sed 's|.*/||')"
$PY -m pip install -q $ROOT/dist/torch-*.whl > $LOG/pip-torch.log 2>&1 || { log "FATAL: torch wheel install failed"; tail -5 $LOG/pip-torch.log; exit 1; }

# vision + audio at pytorch's own CI pins; vision with FORCE_CUDA=1 so its HIP ops are built (as the reference recipe)
for comp in vision audio; do
  pin=$(cat $ROOT/src/pytorch/.github/ci_commit_pins/$comp.txt)
  d=$ROOT/src/$comp
  if [ ! -d $d/.git ]; then
    git clone --depth 1 --recurse-submodules --shallow-submodules --jobs 4 https://github.com/pytorch/$comp.git $d > $LOG/clone-$comp.log 2>&1 \
      || { log "FATAL: clone $comp failed"; exit 1; }
    git -C $d fetch --depth=1 origin "$pin" >> $LOG/clone-$comp.log 2>&1 && git -C $d checkout -q "$pin" >> $LOG/clone-$comp.log 2>&1 \
      || { log "FATAL: checkout $comp $pin failed"; exit 1; }
    git -C $d submodule update --init --recursive --depth 1 >> $LOG/clone-$comp.log 2>&1 || true
  fi
  cd $d
  cver="$(sed -E 's/(a|b|rc)[0-9]+$//' version.txt)"
  log "building $comp ${cver}+$SUFFIX (pin $pin) ..."
  t1=$(date +%s)
  if ! FORCE_CUDA=$([ $comp = vision ] && echo 1 || echo 0) VERSION="$cver" BUILD_VERSION="${cver}+${SUFFIX}" PYTORCH_VERSION="$(cat $ROOT/torch-version.txt)" \
      $PY -m build --wheel --no-isolation --outdir $ROOT/dist > $LOG/build-$comp.log 2>&1; then
    log "FATAL: $comp build failed: $(grep -m2 -iE 'error' $LOG/build-$comp.log | cut -c1-160 | tr '\n' ' ')"; exit 1
  fi
  log "$comp wheel built in $(( $(date +%s)-t1 ))s"
  $PY -m pip install -q --no-deps $ROOT/dist/torch${comp}-*.whl > $LOG/pip-$comp.log 2>&1 || { log "FATAL: $comp wheel install failed"; tail -5 $LOG/pip-$comp.log; exit 1; }
done

log "=== wheels in $ROOT/dist ==="
ls -la $ROOT/dist/*.whl | awk '{print $5" "$9}' | tee -a $LOG/build.log
log "=== torch build done ==="
