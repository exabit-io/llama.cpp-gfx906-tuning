#!/usr/bin/env bash
# launch.sh - the LP-chosen llama-server configurations for Qwen3.8-27B Q8_0 on 4 x gfx906.
#
#   settings/launch.sh <profile> [extra llama-server args]
#
# Builds (env vars, see README section 2 and patches/README.md):
#   LLAMA_PROD   production build (/opt/llama.cpp-prod -> /opt/llama.cpp-mxxm-fh-nq since 2026-09-08): fork tile table + MMVQ patches + the
#                2026-09-08 series (patches/0001-0009: custom-allreduce gate, one-column whole-block load, exact kernel folds); default for every profile
#   LLAMA_PROD0  the 2026-09-07 production binary /opt/llama.cpp-mxxm-fh, kept as the reference for the report numbers
#   LLAMA_STOCK  upstream b10288, the reference build (24-27% slower than production on single dies at 4-8 slots too)
#
# profiles (README section 3, optimize/results.md; figures are the production build unless noted):
#   single   one user, any context to 256K: tp4 + MTP draft 3                      ~74-79 tok/s measured 2026-09-08 (78.5 at 2K, 73.9 at 32K); ~58 without MTP
#   team     up to 16 users, 32K each (84K fits): tp4, 16 slots                     ~196-204 tok/s decode, ~12 per user;
#            server-measured 82-83 tok/s aggregate at 12-16 clients on 1300/256 requests, TTFT 5.6 s
#   busy     many users, 32K per slot: tp4, 16 slots (same launch as team)          server level 2026-09-08: 32 slots give 85.0 tok/s at 16 clients and 80.0 at 32
#            (3.3 per user) against 83.4 on 16 slots — slots beyond 16 add nothing and halve the per-user rate; -np 32 only for offline generation
#   pairs    two tp2 servers (one per MPX module), 8 slots each, ports PORT/PORT+1   production: 91-100 tok/s at 8-16 clients (server level, +20% over tp4 -np 16);
#            the answer to head-of-line blocking: route long prompts to one pair, short to the other
#   long     2 slots x 256K, MTP draft 3 (LP profile H: ~65 tok/s, 32 per stream; the fastest per-stream long-context setting)
#   long8    8 slots x 128K, f16, MTP draft 1 (LP profile G: ~92 tok/s, 12 per stream; M3 2026-09-08)
#   ceiling  8 slots x 160K f16 + MTP draft 1 (29 GiB/die)                        ~81 tok/s, 10.2 per stream
#   batch    four single-die instances, 8 slots each, 4K per slot, ports PORT..PORT+3   ~269 tok/s aggregate (model from the production one-die cells)
#   ingest   prefill-heavy: 12 slots x 32K, micro-batch 4096                        ~990 tok/s prompt reading at 32K
#
# Power: settings/powercap.sh <W> before launching. 200 W max; 185 within 2%; 170 within 5%; 140 within 10%;
# 125 W on the hyperconverged nodes (envelope + economics); caps below ~85 W are accepted and ignored.
# Never load the host CPU beside four pinned dies: DC total over 1228 W latches every die at 1000 MHz until a cold cycle.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-/root/models/Qwen3.8-27B-Q8_0.gguf}
PORT=${PORT:-8089}
HOST=${HOST:-127.0.0.1}
LLAMA_PROD=${LLAMA_PROD:-/opt/llama.cpp-prod/bin}
LLAMA_STOCK=${LLAMA_STOCK:-/opt/llama.cpp/bin}
BIN=${BIN:-$LLAMA_PROD}
# The /opt builds carry no rpath and /etc/ld.so.conf.d/llama.cpp.conf points at the stock /opt/llama.cpp/lib, so without this
# line the production binary silently loads the STOCK libggml-hip (verified with ldd, 2026-09-08). Every report number was
# measured with LD_LIBRARY_PATH set to the build's own lib/.
export LD_LIBRARY_PATH="$(dirname "$BIN")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

set -a; . "$HERE/gfx906.env"; set +a
# The custom-allreduce settings need the GGML_TP_AR_MAX_NE size gate (fusion tree 25e1d46+). On a binary without it the
# fork's default gate would cost 8% at 8-16 slots, so drop them there (2026-09-08 gate sweep).
if ! strings "$(dirname "$BIN")/lib/libggml-hip.so.0" 2>/dev/null | grep -q GGML_TP_AR_MAX_NE; then
  unset GGML_ENABLE_CUSTOM_AR GGML_TP_AR_MAX_NE
  echo "launch.sh: $BIN has no GGML_TP_AR_MAX_NE knob; custom allreduce left off" >&2
fi

COMMON=(-m "$MODEL" -fa on -b 2048 -cb --host "$HOST")
TP4=(--device rocm0,rocm1,rocm2,rocm3 -sm tensor)

case "${1:-}" in
  single)
    shift
    exec "$BIN/llama-server" "${COMMON[@]}" "${TP4[@]}" -np 1 -c 262144 -ub 2048 \
      --spec-type draft-mtp --spec-draft-n-max 3 --port "$PORT" "$@"
    ;;
  team)
    shift
    # 16 slots on the production build (fast-path MMVQ to 16 columns). On the stock build use -np 8: 9-15 slots cost a full 16-wide MMQ tile.
    exec "$BIN/llama-server" "${COMMON[@]}" "${TP4[@]}" -np 16 -c $((16*32768)) -ub 2048 --port "$PORT" "$@"
    ;;
  busy)
    shift
    # 16 slots, as team: 32 slots are +4-5% on the decode-only bench (M2) but 80-85 tok/s against 83 at the server level with half the
    # per-user rate (server_final_prod, 2026-09-08). Offline generation that wants 32 slots: -np 32 -c $((32*32768)) by hand.
    exec "$BIN/llama-server" "${COMMON[@]}" "${TP4[@]}" -np 16 -c $((16*32768)) -ub 2048 --port "$PORT" "$@"
    ;;
  pairs)
    shift
    "$BIN/llama-server" "${COMMON[@]}" --device rocm0,rocm1 -sm tensor -np 8 -c $((8*65536)) -ub 2048 --port "$PORT" "$@" \
      > /tmp/llama-server-pair0.log 2>&1 &
    "$BIN/llama-server" "${COMMON[@]}" --device rocm2,rocm3 -sm tensor -np 8 -c $((8*65536)) -ub 2048 --port $((PORT+1)) "$@" \
      > /tmp/llama-server-pair1.log 2>&1 &
    wait
    ;;
  long)
    shift
    exec "$BIN/llama-server" "${COMMON[@]}" "${TP4[@]}" -np 2 -c $((2*262144)) -ub 2048 \
      --spec-type draft-mtp --spec-draft-n-max 3 --port "$PORT" "$@"
    ;;
  long8)
    shift
    # M3 (2026-09-08): 8 slots x draft 1 is +5% at 32K on the production build (16-row verify batches run in the 16-column MMVQ); -2% at 2K.
    exec "$BIN/llama-server" "${COMMON[@]}" "${TP4[@]}" -np 8 -c $((8*131072)) -ub 2048 --spec-type draft-mtp --spec-draft-n-max 1 --port "$PORT" "$@"
    ;;
  ceiling)
    shift
    exec "$BIN/llama-server" "${COMMON[@]}" "${TP4[@]}" -np 8 -c $((8*163840)) -ub 2048 --spec-type draft-mtp --spec-draft-n-max 1 --port "$PORT" "$@"
    ;;
  ingest)
    shift
    exec "$BIN/llama-server" "${COMMON[@]}" "${TP4[@]}" -np 12 -c $((12*32768)) -ub 4096 --port "$PORT" "$@"
    ;;
  batch)
    shift
    # one server per die on ports PORT..PORT+3; no RCCL involved, so the NCCL_* variables are inert.
    for i in 0 1 2 3; do
      "$BIN/llama-server" "${COMMON[@]}" --device "rocm$i" -np 8 -c $((8*4096)) -ub 2048 --port $((PORT+i)) "$@" \
        > "/tmp/llama-server-rocm$i.log" 2>&1 &
    done
    wait
    ;;
  *)
    sed -n 2,25p "$0"; exit 1
    ;;
esac
