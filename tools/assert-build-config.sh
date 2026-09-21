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
a=$(strings "$bd"/bin/libggml-hip.so* 2>/dev/null | grep -cx 'GGML_TP_AR_MAX_NE')
[ "${a:-0}" -gt 0 ] && say OK "AR size gate knob compiled in" || say WARN "GGML_TP_AR_MAX_NE not compiled in — setting it in env will do nothing"
exit $fail
