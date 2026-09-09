#!/bin/bash
# TODO 9 / NEXT-STEPS S4 diagnostic: hardware counters on the flash-attention kernels at head size 256 (production build, tp4; per-die counters are
# identical across the lanes): (a) prefill 4 x 32K, (b) decode at 128K depth (llama-bench -d). Also a kernel-trace stats pass for the FA share.
set -u
B=/root/rocm-tests/bench; M=/root/models/Qwen3.8-27B-Q8_0.gguf; TAG=qwen38-27b-fa-counters; P=/opt/llama.cpp-prod
. $B/gpu-test-env.sh
PROG=$B/$TAG.progress; echo $$ > $B/$TAG.pid; : > $PROG
log() { echo "$(date -Is) $*" >> $PROG; }
cleanup() { trap - INT TERM; pkill -P $$ 2>/dev/null; kill ${SAMP:-} 2>/dev/null; restore; log KILLED; exit 1; }
trap cleanup INT TERM
setmax; start_sampler $B/$TAG-clocks.txt; log "START $(state_line)"
export NCCL_TOPO_FILE=/root/rccl_topo_fixed.xml NCCL_MIN_NCHANNELS=16 LD_LIBRARY_PATH=$P/lib
D4="--device rocm0,rocm1,rocm2,rocm3 -sm tensor"; OUT=$B/$TAG.md; T=$B/trace-fa; rm -rf $T; mkdir -p $T
echo "# $TAG  $(date -Is)  FA head-256 counters, production build tp4" > $OUT
PMC="--pmc SQ_INSTS_VALU SQ_BUSY_CYCLES SQ_WAIT_INST_ANY TCP_TOTAL_CACHE_ACCESSES"
log "prefill counters: batched-bench tp4 -npl 4 -npp 32768 -ntg 1"
timeout 1500 rocprofv3 $PMC --kernel-trace --stats -f csv -d $T/prefill -o pre -- $P/bin/llama-batched-bench -m $M $D4 -fa on -b 2048 -ub 2048 -c 135168 -npp 32768 -ntg 1 -npl 4 > $B/$TAG-prefill.md 2>$B/$TAG-prefill.err; log "prefill rc $?"
log "decode counters: llama-bench tp4 -d 131072 -n 32"
timeout 1800 rocprofv3 $PMC --kernel-trace --stats -f csv -d $T/decode -o dec -- $P/bin/llama-bench -m $M --device rocm0/rocm1/rocm2/rocm3 -sm tensor -fa 1 -p 0 -n 32 -d 131072 -r 1 -o md > $B/$TAG-decode.md 2>$B/$TAG-decode.err; log "decode rc $?"
python3 - "$T" >> $OUT <<'PY'
import csv, glob, sys, collections, os
T = sys.argv[1]
for phase in ('prefill', 'decode'):
    files = glob.glob(f'{T}/{phase}/**/*counter_collection.csv', recursive=True)
    print(f'\n## {phase}: counter files {len(files)}')
    if not files: continue
    agg = collections.defaultdict(lambda: collections.defaultdict(float)); n = collections.Counter()
    for fn in files:
        for r in csv.DictReader(open(fn)):
            k = r.get('Kernel_Name', '')[:70]; c = r.get('Counter_Name'); v = float(r.get('Counter_Value', 0) or 0)
            if r.get('Agent_Id', '') not in ('', 'Agent 1', '1'): pass
            agg[k][c] += v; n[k] += 1
    rows = []
    for k, d in agg.items():
        if 'flash_attn' in k or 'mul_mat' in k:
            busy = d.get('SQ_BUSY_CYCLES', 0); valu = d.get('SQ_INSTS_VALU', 0); wait = d.get('SQ_WAIT_INST_ANY', 0); tcp = d.get('TCP_TOTAL_CACHE_ACCESSES', 0)
            rows.append((busy, k, valu, wait, tcp))
    rows.sort(reverse=True)
    print('| kernel | SQ_BUSY_CYCLES | SQ_INSTS_VALU | VALU per busy cycle | SQ_WAIT_INST_ANY / busy | TCP accesses |'); print('|---|---:|---:|---:|---:|---:|')
    for busy, k, valu, wait, tcp in rows[:8]:
        print(f'| {k} | {busy:.3g} | {valu:.3g} | {valu/busy if busy else 0:.3f} | {wait/busy if busy else 0:.3f} | {tcp:.3g} |')
PY
{ echo; echo "## kernel-time share (from the kernel stats of each pass)"; for ph in prefill decode; do f=$(ls $T/$ph/*/*kernel_stats.csv $T/$ph/*kernel_stats.csv 2>/dev/null | head -1); echo "### $ph"; [ -n "$f" ] && head -12 "$f" | cut -c1-160; done; } >> $OUT
echo "# done $(date -Is)" >> $OUT
kill $SAMP 2>/dev/null; restore; log ALLDONE; touch $B/.$TAG-done
