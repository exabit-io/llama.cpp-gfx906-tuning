#!/bin/bash
# bonsai-check.sh -- does the installed package (mx-llama-cpp-gfx906 1.1.0, llama.cpp v1.1.0 = b12774) run
# Ternary-Bonsai-2.5-27B-Q2_0.gguf correctly on gfx906, and in which device modes. Functional check, not a timing:
# no rate printed here is a result (REQUIREMENTS R2.2: smoke tests may use any depth, their numbers are never quoted).
#
# What the model needs (its card, read 2026-11-28): the weights are ternary in a Hadamard-rotated basis stored as
# llama.cpp's Q2_0 type; the runtime must apply the matching transform to the activations (upstream since b11556).
# "Older builds load Q2_0 without any warning and produce garbage". Our tree has the type, MMQ and MMVQ kernels for
# it in the GPU backend (gfx906 takes upstream's GCN tile table), and an FWHT kernel. Not known before this run:
# whether the fork's tensor split computes it (its repack path lists Q8_0, MXFP4, IQ4_NL, Q4_K, Q5_K, Q6_K, Q5_1 and
# not Q2_0), and whether any operation falls back to the CPU.
#
# Modes: b2s one die (--device rocm0); b2l four dies, -sm layer; b2t four dies, -sm tensor (as the service runs).
# Steps per mode: (1) greedy text, 96 tokens, the project's prompt; (2) perplexity on wikitext-2 test with the project's
# command (-c 16384 --chunks 6 -b 2048 -ub 2048: Qwen3.8-27B Q8_0 reads 5.6088 +/- 0.0622 on this package).
#
# READING, WRITTEN BEFORE THE FIRST RUN.
#   (S1) Loads: every layer offloaded in each mode; the log's tensor-type list is reported (how much of the file is
#        Q2_0); "graph splits" is reported per mode next to the same line of a Qwen Q8_0 log of this package.
#        A mode that does not load, or aborts, is reported with its last log lines and is not timed.
#   (S2) The runtime transform works if one-die perplexity is under 20 (garbage reads in the hundreds or more; the
#        Q8_0 model reads 5.61; a ternary model is expected somewhat above that). Over 20: stop, nothing is timed.
#   (S3) The three modes compute the same model: their perplexities must lie within 0.5 % of the one-die value
#        (rounding between modes is far smaller, a wrong path far larger). A mode outside that does not compute this
#        model correctly on our tree: reported as a defect, not timed.
#   (S4) The greedy texts are printed; modes may differ by rounding. Whether the text is sensible is read by eye and
#        said as such.
set -u
W=/root/bonsai-bench; R=/root/rocm-tests/bench; O=$W/check; mkdir -p $O
PKG=/opt/mx-llama.cpp; MB=/root/models/Ternary-Bonsai-2.5-27B-Q2_0.gguf; F=/root/models/wikitext-2-raw/wiki.test.raw
D4="--device rocm0,rocm1,rocm2,rocm3"
P=$O/progress; DONE=$O/.done; rm -f $DONE
log(){ echo "$(date -Is) [check] $*" | tee -a $P; }
kpid(){ [ -n "${1:-}" ] && [ "${1:-0}" -gt 1 ] 2>/dev/null && kill "$1" 2>/dev/null; return 0; }
devs(){ case $1 in b2s) echo "--device rocm0";; b2l) echo "$D4 -sm layer";; b2t) echo "$D4 -sm tensor";; esac; }
[ -f $MB ] || { log "FATAL: $MB missing"; exit 1; }
[ -f $F ] || { log "FATAL: $F missing"; exit 1; }
for p in llama-server llama-batched-b llama-perplexit llama-completi test-backend-op llama-cli cmake ninja; do
  pgrep -x $p >/dev/null && { log "FATAL: $p is running"; exit 1; }
done
log "package $(dpkg-query -W -f='${Version}' mx-llama-cpp-gfx906), source $(cut -c1-9 /usr/share/doc/mx-llama-cpp-gfx906/SOURCE-COMMIT); model $(stat -c %s $MB) bytes"
. $R/gpu-test-env.sh
export GGML_CUDA_ALLREDUCE=nccl GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1
BPID=""
cleanup(){ trap - INT TERM EXIT; kpid "${BPID:-}"; kpid "${SAMP:-}"; kpid "${WDOG:-}"; kpid "${SMCL:-}"; restore; log "STOPPED (trap)"; touch $DONE; exit 1; }
trap cleanup INT TERM EXIT
setmax; start_sampler $O/clocks.txt
QUEUE_PID=$$ DONEFLAG=$DONE SMCLOG=$O/smc.log SCLK_GUARD=1 nohup $R/clamp-watchdog-v2.sh >> $O/watchdog.out 2>&1 &
WDOG=$!
nohup $R/smc-log.sh $O/smc.log >/dev/null 2>&1 &
SMCL=$!
sleep 6
log "$(state_line)"
run(){ # run NAME TIMEOUT cmd...
  local name=$1 to=$2; shift 2
  env -u LD_LIBRARY_PATH timeout $to "$@" > $O/$name.out 2> $O/$name.log < /dev/null &
  BPID=$!; wait $BPID; local rc=$?; BPID=""; return $rc
}
for m in ${MODES:-b2s b2l b2t}; do
  run $m-text 900 $PKG/bin/llama-completion -m $MB $(devs $m) -ngl all -fa on -c 4096 -n 96 --temp 0 --top-k 1 -s 0 -no-cnv --no-warmup -p "Explain in two sentences why the sky is blue."
  rc=$?
  log "$m text rc=$rc, $(wc -c < $O/$m-text.out) bytes, md5 $(md5sum < $O/$m-text.out | cut -c1-12): $(tr '\n' ' ' < $O/$m-text.out | cut -c1-420)"
  log "$m load: $(grep -oE 'offloaded [0-9]+/[0-9]+ layers to GPU' $O/$m-text.log | tail -1); $(grep -E 'graph splits' $O/$m-text.log | tail -1 | sed 's/^.*: *//' | cut -c1-80); types: $(grep -E 'llama_model_loader: - type' $O/$m-text.log | sed 's/.*- type *//' | tr -s ' ' | tr '\n' ';' | cut -c1-200)"
  [ $rc = 0 ] || log "$m text FAILED: $(tail -4 $O/$m-text.log | tr '\n' ' ' | cut -c1-400)"
  run $m-ppl 3600 $PKG/bin/llama-perplexity -m $MB $(devs $m) -ngl all -fa on -b 2048 -ub 2048 -c 16384 --chunks 6 -f $F
  rc=$?
  log "$m perplexity rc=$rc: $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $O/$m-ppl.log $O/$m-ppl.out 2>/dev/null | tail -1 | sed 's/^[^:]*://'); $(grep -E 'graph splits' $O/$m-ppl.log | tail -1 | sed 's/^.*: *//' | cut -c1-80); warnings: $(grep -ciE 'no FlashAttention vector kernel|falling back|not supported' $O/$m-ppl.log)"
  [ $rc = 0 ] || log "$m perplexity FAILED: $(tail -4 $O/$m-ppl.log | tr '\n' ' ' | cut -c1-400)"
done
trap - INT TERM EXIT
kpid $SAMP; kpid $SMCL; touch $DONE; sleep 6; kpid $WDOG; restore
python3 -I - $O <<'PY' | tee -a $P
import re, sys, os
o = sys.argv[1]; v = {}
for m in ('b2s', 'b2l', 'b2t'):
    for fn in (m + '-ppl.log', m + '-ppl.out'):
        p = os.path.join(o, fn)
        if os.path.exists(p):
            x = re.findall(r'Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)', open(p, errors='replace').read())
            if x: v[m] = (float(x[-1][0]), float(x[-1][1]))
print('[check] perplexity: ' + '; '.join('%s %.4f +/- %.4f' % (m, a, b) for m, (a, b) in v.items()))
if 'b2s' in v:
    base = v['b2s'][0]
    print('[check] (S2) one die %.4f: %s' % (base, 'under 20, the runtime transform works' if base < 20 else 'OVER 20: garbage, nothing is timed'))
    for m in ('b2l', 'b2t'):
        if m in v:
            d = (v[m][0] / base - 1) * 100
            print('[check] (S3) %s against one die: %+.3f %% -> %s' % (m, d, 'same model' if abs(d) <= 0.5 else 'NOT the same computation: defect, not timed'))
        else:
            print('[check] (S3) %s: no perplexity (mode failed)' % m)
else:
    print('[check] (S2) no one-die perplexity')
PY
log "=== BONSAI-CHECK DONE ==="
