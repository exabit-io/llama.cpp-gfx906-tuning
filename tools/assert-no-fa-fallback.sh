#!/bin/bash
# assert-no-fa-fallback.sh LOGFILE... — fail if any run fell back to f16 KV conversion.
#
# llama.cpp warns when a KV type pair has no compiled FlashAttention kernel:
#
#   ggml_cuda_flash_attn_ext_vec: no FlashAttention vector kernel compiled for K/V types
#   q8_0-q5_1, converting K and V to f16 instead (slow). Add "q8_0-q5_1" to GGML_CUDA_FA_QUANTS
#
# It is NOT silent -- it says exactly what happened and how to fix it. I spent 2026-09-22 asserting
# the opposite from an inference, while the warning sat in the log I had already captured. Any cell
# whose log contains this line measured an f16-conversion path, not the quantised kernel, and its
# number is not what it claims to be.
set -u
[ $# -ge 1 ] || { echo "usage: assert-no-fa-fallback.sh LOGFILE..." >&2; exit 2; }
fail=0
for f in "$@"; do
  [ -f "$f" ] || continue
  n=$(grep -c 'no FlashAttention vector kernel compiled' "$f" 2>/dev/null)
  if [ "${n:-0}" -gt 0 ]; then
    kv=$(grep -oE 'K/V types [a-z0-9_]+-[a-z0-9_]+' "$f" | head -1 | awk '{print $3}')
    echo "FAIL  $(basename "$f"): fell back to f16 conversion for K/V ${kv:-unknown} ($n warning(s))"
    fail=1
  fi
done
[ $fail -eq 0 ] && echo "OK    no FA fallback in $# log(s)"
exit $fail
