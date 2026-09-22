#!/bin/bash
# assert-build-config.sh BUILD_DIR — refuse a build that violates a recorded build requirement.
#
# R3.11 (lead, 2026-09-21): every build ships with GGML_HIP_RCCL=ON. Upstream's default is OFF and this
# campaign inherited it without audit, which silently put every AllReduce measurement on the
# meta-backend butterfly path instead of RCCL. A requirement that lives only in prose has already
# failed once here (see survey-lint.py's header), so this is the enforcement.
set -u
bd=${1:?usage: assert-build-config.sh BUILD_DIR}
fail=0
say(){ printf '  %-9s %s\n' "$1" "$2"; }
[ -x "$bd/bin/llama-batched-bench" ] || { say FAIL "$bd has no llama-batched-bench"; exit 1; }
n=$(strings "$bd"/bin/libggml-hip.so* 2>/dev/null | grep -cE 'ncclCommInit|ncclAllReduce')
if [ "${n:-0}" -gt 0 ]; then say OK "RCCL linked ($n symbols) — R3.11 satisfied"
else say FAIL "NO RCCL symbols — built with GGML_HIP_RCCL=OFF, violates R3.11"; fail=1; fi
if [ -f "$bd/CMakeCache.txt" ]; then
  v=$(grep -E '^GGML_HIP_RCCL:BOOL=' "$bd/CMakeCache.txt" | cut -d= -f2)
  [ "${v:-}" = "ON" ] && say OK "CMakeCache GGML_HIP_RCCL=ON" || { say FAIL "CMakeCache GGML_HIP_RCCL=${v:-unset}"; fail=1; }
fi
# FA_QUANTS must match the campaign value, or a build differs from its comparison arm in the set of
# compiled attention kernels as well as in whatever is under test. An uncompiled combination does NOT
# fail loudly -- it silently takes a slower generic path (measured 2026-09-22: q5_1 ran 8% slow) -- so
# a mismatch here is invisible at runtime and corrupts the comparison quietly.
WANT='all'   # R3.11 (lead, 2026-09-22): compile every K/V combination. A missing kernel does not
             # fail, it converts K and V to f16 and warns -- so an explicit subset is the fragile
             # choice, not the careful one. Cost measured at 16 MiB of .so and a few ccache-warm minutes.
if [ -f "$bd/CMakeCache.txt" ]; then
  fq=$(grep -E '^GGML_CUDA_FA_QUANTS:STRING=' "$bd/CMakeCache.txt" | cut -d= -f2-)
  if [ "$fq" = "$WANT" ]; then say OK "FA_QUANTS=all — every K/V combination compiled (R3.11)"
  else say FAIL "FA_QUANTS='$fq' but R3.11 requires 'all' — an uncompiled combo silently converts to f16"; fail=1; fi
fi
a=$(strings "$bd"/bin/libggml-hip.so* 2>/dev/null | grep -cx 'GGML_TP_AR_MAX_NE')
[ "${a:-0}" -gt 0 ] && say OK "AR size gate knob compiled in" || say WARN "GGML_TP_AR_MAX_NE not compiled in — setting it in env will do nothing"
exit $fail
